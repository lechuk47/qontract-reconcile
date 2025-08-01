from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Literal
from urllib.parse import urlparse

from gitlab.const import PipelineStatus
from gitlab.v4.objects import ProjectMergeRequest

from reconcile.typed_queries.github_orgs import GithubOrgV1
from reconcile.typed_queries.gitlab_instances import GitlabInstanceV1
from reconcile.utils.github_api import GithubRepositoryApi
from reconcile.utils.gitlab_api import (
    GitLabApi,
    MRState,
)
from reconcile.utils.mr.base import MergeRequestBase
from reconcile.utils.secret_reader import (
    HasSecret,
    SecretReaderBase,
)

GITHUB_BASE_URL = "https://github.com/"

Platform = Literal["github", "gitlab"]

SUPPORTED_PLATFORMS: list[Platform] = ["github", "gitlab"]


@dataclass(frozen=True)
class RepoInfo:
    platform: Platform | None
    name: str


class MRCheckStatus(Enum):
    NONE = 0
    SUCCESS = 1
    FAILED = 2
    RUNNING = 3


@dataclass(order=True)
class Commit:
    repo: str
    sha: str
    date: datetime


class VCSMissingSourceBranchError(Exception):
    pass


class VCS:
    """
    Abstraction layer for aggregating different Version Control Systems.
    The idea is to abstract away the differences between
    Gitlab and Github for fetching publisher state.
    Further this acts as a wrapper around our gitlab client
    for interactions with the app-interface repo. That makes
    setting up tests easier.
    """

    def __init__(
        self,
        secret_reader: SecretReaderBase,
        github_orgs: Iterable[GithubOrgV1],
        gitlab_instances: Iterable[GitlabInstanceV1],
        app_interface_repo_url: str,
        dry_run: bool,
        allow_deleting_mrs: bool = False,
        allow_opening_mrs: bool = False,
        gitlab_instance: GitLabApi | None = None,
        default_gh_token: str | None = None,
        app_interface_api: GitLabApi | None = None,
        github_api_per_repo_url: dict[str, GithubRepositoryApi] | None = None,
    ):
        self._dry_run = dry_run
        self._allow_deleting_mrs = allow_deleting_mrs
        self._allow_opening_mrs = allow_opening_mrs
        self._secret_reader = secret_reader
        self._gh_per_repo_url: dict[str, GithubRepositoryApi] = (
            github_api_per_repo_url or {}
        )
        self._default_gh_token = default_gh_token or self._get_default_gh_token(
            github_orgs=github_orgs
        )
        self._gitlab_instance = gitlab_instance or self._gitlab_api(
            gitlab_instances=gitlab_instances
        )
        self._app_interface_api = app_interface_api or self._init_app_interface_api(
            gitlab_instances=gitlab_instances,
            app_interface_repo_url=app_interface_repo_url,
        )
        self._is_commit_sha_regex = re.compile(r"^[0-9a-f]{40}$")

    def _get_default_gh_token(
        self,
        github_orgs: Iterable[GithubOrgV1],
    ) -> str:
        defaults: list[str] = []
        for org in github_orgs:
            if not org.default:
                continue
            token = self._secret_reader.read_secret(org.token)
            defaults.append(token)
        if len(defaults) == 0:
            raise RuntimeError("No default GitHub token found.")
        if len(defaults) > 1:
            raise RuntimeError("More than 1 default token for GitHub found.")
        return defaults[0]

    def _init_github(
        self, repo_url: str, auth_code: HasSecret | None
    ) -> GithubRepositoryApi:
        if repo_url not in self._gh_per_repo_url:
            if auth_code:
                token = self._secret_reader.read_secret(auth_code)
            else:
                token = self._default_gh_token
            self._gh_per_repo_url[repo_url] = self._github_api(
                token=token, repo_url=repo_url
            )
        return self._gh_per_repo_url[repo_url]

    def _github_api(self, token: str, repo_url: str) -> GithubRepositoryApi:
        return GithubRepositoryApi(repo_url=repo_url, token=token)

    def _gitlab_api(
        self,
        gitlab_instances: Iterable[GitlabInstanceV1],
    ) -> GitLabApi:
        return GitLabApi(
            next(iter(gitlab_instances)).dict(by_alias=True),
            secret_reader=self._secret_reader,
        )

    def _init_app_interface_api(
        self,
        gitlab_instances: Iterable[GitlabInstanceV1],
        app_interface_repo_url: str,
    ) -> GitLabApi:
        return GitLabApi(
            next(iter(gitlab_instances)).dict(by_alias=True),
            secret_reader=self._secret_reader,
            project_url=app_interface_repo_url,
        )

    def get_gitlab_mr_check_status(self, mr: ProjectMergeRequest) -> MRCheckStatus:
        pipelines = self._gitlab_instance.get_merge_request_pipelines(mr)
        if not pipelines:
            return MRCheckStatus.NONE
        last_pipeline_result = pipelines[0].status
        match last_pipeline_result:
            case PipelineStatus.SUCCESS:
                return MRCheckStatus.SUCCESS
            case PipelineStatus.RUNNING:
                return MRCheckStatus.RUNNING
            case PipelineStatus.FAILED:
                return MRCheckStatus.FAILED
            case _:
                # Lets assume all other states as non-present
                return MRCheckStatus.NONE

    @staticmethod
    def parse_repo_url(url: str) -> RepoInfo:
        """
        Parse a repository URL and return a RepoInfo object.
        `platform` can be 'github', 'gitlab' or None if not recognized,
        it's inferred from the URL host.
        `name` is the path part of the URL, stripped of leading and trailing slashes.
        """
        parsed_url = urlparse(url)
        platform = next(
            (
                p
                for p in SUPPORTED_PLATFORMS
                if (hostname := parsed_url.hostname) and p in hostname
            ),
            None,
        )
        name = parsed_url.path.strip("/").removesuffix(".git")
        return RepoInfo(platform=platform, name=name)

    def get_commit_sha(
        self, repo_url: str, ref: str, auth_code: HasSecret | None
    ) -> str:
        if bool(self._is_commit_sha_regex.search(ref)):
            return ref
        repo_info = self.parse_repo_url(repo_url)
        match repo_info.platform:
            case "github":
                github = self._init_github(repo_url=repo_url, auth_code=auth_code)
                return github.get_commit_sha(ref=ref)
            case "gitlab":
                return self._gitlab_instance.get_commit_sha(ref=ref, repo_url=repo_url)
            case _:
                raise ValueError(f"Unsupported repository URL: {repo_url}")

    def get_commits_between(
        self,
        repo_url: str,
        commit_from: str,
        commit_to: str,
        auth_code: HasSecret | None,
    ) -> list[Commit]:
        """
        Return a list of commits between two commits.
        Note, that the commit_to is included in the result list, whereas commit_from is not included.
        """
        repo_info = self.parse_repo_url(repo_url)
        match repo_info.platform:
            case "github":
                github = self._init_github(repo_url=repo_url, auth_code=auth_code)
                data = github.compare(commit_from=commit_from, commit_to=commit_to)
                return [
                    Commit(
                        repo=repo_url,
                        sha=gh_commit.sha,
                        date=gh_commit.commit.committer.date,
                    )
                    for gh_commit in data
                ]
            case "gitlab":
                data = self._gitlab_instance.repository_compare(
                    repo_url=repo_url, ref_from=commit_from, ref_to=commit_to
                )
                return [
                    Commit(
                        repo=repo_url,
                        sha=gl_commit["id"],
                        date=datetime.fromisoformat(gl_commit["committed_date"]),
                    )
                    for gl_commit in data
                ]
            case _:
                raise ValueError(f"Unsupported repository URL: {repo_url}")

    def close_app_interface_mr(self, mr: ProjectMergeRequest, comment: str) -> None:
        if not self._allow_deleting_mrs:
            logging.info("Deleting MRs is disabled. Skipping.")
        if not self._dry_run and self._allow_deleting_mrs:
            self._app_interface_api.add_comment_to_merge_request(
                merge_request=mr,
                body=comment,
            )
            source_branch = mr.attributes.get("source_branch")
            if not source_branch:
                raise VCSMissingSourceBranchError(
                    f"Source branch is missing for MR {mr.attributes.get('iid')}"
                )
            self._app_interface_api.close(mr)
            self._app_interface_api.delete_branch(source_branch)

    def get_file_content_from_app_interface_ref(
        self, file_path: str, ref: str = "master", is_data: bool = True
    ) -> str:
        if is_data:
            file_path = (
                f"data/{file_path.lstrip('/')}"
                if not file_path.startswith("data")
                else file_path
            )
        file = self._app_interface_api.get_raw_file(
            project=self._app_interface_api.project,
            path=file_path,
            ref=ref,
        )
        return file.decode("utf-8")

    def get_open_app_interface_merge_requests(self) -> list[ProjectMergeRequest]:
        return self._app_interface_api.get_merge_requests(state=MRState.OPENED)

    def open_app_interface_merge_request(self, mr: MergeRequestBase) -> None:
        if not self._allow_opening_mrs:
            logging.info("Creating MRs is disabled. Skipping.")
        if not self._dry_run and self._allow_opening_mrs:
            mr.submit_to_gitlab(gitlab_cli=self._app_interface_api)

    def cleanup(self) -> None:
        for gh_client in self._gh_per_repo_url.values():
            gh_client.cleanup()
        self._gitlab_instance.cleanup()
        self._app_interface_api.cleanup()
