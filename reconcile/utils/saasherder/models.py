import base64
import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, NotRequired, TypedDict

from github import Github
from pydantic import (
    BaseModel,
    Field,
)

from reconcile.gql_definitions.fragments.saas_slo_document import (
    SLODocument,
)
from reconcile.utils.jenkins_api import JobBuildState
from reconcile.utils.oc_connection_parameters import Cluster
from reconcile.utils.saasherder.interfaces import (
    HasParameters,
    HasSecretParameters,
    ManagedResourceName,
    SaasApp,
    SaasEnvironment,
    SaasFile,
    SaasPipelinesProviders,
    SaasResourceTemplate,
    SaasResourceTemplateTarget,
)
from reconcile.utils.secret_reader import SecretReaderBase


class Providers(Enum):
    TEKTON = "tekton"


class TriggerTypes:
    CONFIGS = 0
    MOVING_COMMITS = 1
    UPSTREAM_JOBS = 2
    CONTAINER_IMAGES = 3


@dataclass
class UpstreamJob:
    instance: str
    job: str

    def __str__(self) -> str:
        return f"{self.instance}/{self.job}"

    def __repr__(self) -> str:
        return self.__str__()


@dataclass(frozen=True)
class TriggerSpecBase:
    saas_file_name: str
    env_name: str
    timeout: str | None
    pipelines_provider: SaasPipelinesProviders
    resource_template_name: str
    cluster_name: str
    namespace_name: str
    state_content: Any
    reason: str | None
    target_ref: str

    @property
    def state_key(self) -> str:
        raise NotImplementedError("implement this function in inheriting classes")


@dataclass(frozen=True)
class SLOKey:
    slo_document_name: str
    namespace_name: str
    cluster_name: str


class TriggerSpecConfigStateContentNamespaceApp(TypedDict):
    name: str


class TriggerSpecConfigStateContentNamespaceCluster(TypedDict):
    name: str
    serverUrl: str


class TriggerSpecConfigStateContentNamespace(TypedDict):
    name: str
    cluster: TriggerSpecConfigStateContentNamespaceCluster
    app: TriggerSpecConfigStateContentNamespaceApp


class TriggerSpecConfigStateContent(TypedDict):
    """
    dict representation of reconcile.typed_queries.saas_files.SaasResourceTemplateTarget
    with some additional fields.
    """

    path: str | None
    name: str | None
    namespace: TriggerSpecConfigStateContentNamespace
    ref: str | None
    promotion: dict | None
    parameters: str | None
    secretParameters: list[dict] | None
    slos: list[Any] | None
    upstream: Any | None
    images: list[Any] | None
    disable: bool | None
    delete: bool | None

    # additional fields
    saas_file_parameters: str | None
    saas_file_managed_resource_types: list[str]
    saas_file_managed_resource_names: NotRequired[list[Any]]
    url: str
    rt_parameters: str | None
    rt_secretparameters: NotRequired[list[dict]]
    saas_file_secretparameters: NotRequired[list[dict]]


@dataclass(frozen=True)
class TriggerSpecConfig(TriggerSpecBase):
    state_content: TriggerSpecConfigStateContent | None
    resource_template_url: str
    slos: list[SLODocument] | None = None
    target_name: str | None = None

    @property
    def state_key(self) -> str:
        key = (
            f"{self.saas_file_name}/{self.resource_template_name}/{self.cluster_name}/"
            f"{self.namespace_name}/{self.env_name}"
        )
        if self.target_name:
            key += f"/{self.target_name}"
        return key

    def extract_slo_keys(self) -> list[SLOKey]:
        return [
            SLOKey(
                slo_document_name=slo_document.name,
                namespace_name=namespace.namespace.name,
                cluster_name=namespace.namespace.cluster.name,
            )
            for slo_document in self.slos or []
            for namespace in slo_document.namespaces
            if namespace.namespace.name == self.namespace_name
            and namespace.namespace.cluster.name == self.cluster_name
        ]


@dataclass(frozen=True)
class TriggerSpecMovingCommit(TriggerSpecBase):
    state_content: str
    ref: str

    @property
    def state_key(self) -> str:
        key = (
            f"{self.saas_file_name}/{self.resource_template_name}/{self.cluster_name}/"
            f"{self.namespace_name}/{self.env_name}/{self.ref}"
        )
        return key


@dataclass(frozen=True)
class TriggerSpecUpstreamJob(TriggerSpecBase):
    state_content: JobBuildState
    instance_name: str
    job_name: str

    @property
    def state_key(self) -> str:
        key = (
            f"{self.saas_file_name}/{self.resource_template_name}/{self.cluster_name}/"
            f"{self.namespace_name}/{self.env_name}/{self.instance_name}/{self.job_name}"
        )
        return key


@dataclass(frozen=True)
class TriggerSpecContainerImage(TriggerSpecBase):
    state_content: str
    images: Sequence[str]

    @property
    def state_key(self) -> str:
        image_key = "/".join(sorted(self.images))
        key = (
            f"{self.saas_file_name}/{self.resource_template_name}/{self.cluster_name}/"
            f"{self.namespace_name}/{self.env_name}/{image_key}"
        )
        return key


TriggerSpecUnion = (
    TriggerSpecConfig
    | TriggerSpecMovingCommit
    | TriggerSpecUpstreamJob
    | TriggerSpecContainerImage
)


class Namespace(BaseModel):
    name: str
    environment: SaasEnvironment
    app: SaasApp
    cluster: Cluster
    managed_resource_types: list[str] = Field(..., alias="managedResourceTypes")
    managed_resource_names: Sequence[ManagedResourceName] | None = Field(
        ..., alias="managedResourceNames"
    )

    class Config:
        arbitrary_types_allowed = True
        allow_population_by_field_name = True


class PromotionChannelData(BaseModel):
    q_type: str = Field(..., alias="type")

    class Config:
        allow_population_by_field_name = True


class ParentSaasPromotion(BaseModel):
    q_type: str = Field(..., alias="type")
    parent_saas: str | None
    target_config_hash: str | None

    class Config:
        allow_population_by_field_name = True


class PromotionData(BaseModel):
    channel: str | None
    data: list[ParentSaasPromotion | PromotionChannelData] | None = None


class Channel(BaseModel):
    name: str
    publisher_uids: list[str]


class Promotion(BaseModel):
    """Implementation of the SaasPromotion interface for saasherder and AutoPromoter."""

    url: str
    commit_sha: str
    saas_file: str
    target_config_hash: str
    saas_target_uid: str
    soak_days: int
    auto: bool | None = None
    publish: list[str] | None = None
    subscribe: list[Channel] | None = None
    promotion_data: list[PromotionData] | None = None
    saas_file_paths: list[str] | None = None
    target_paths: list[str] | None = None


@dataclass
class ImageAuth:
    username: str | None = None
    password: str | None = None
    auth_server: str | None = None
    docker_config: dict[str, dict[str, dict[str, str]]] | None = None

    def get_docker_config_json(self) -> dict:
        if self.docker_config:
            return self.docker_config
        else:
            return {
                "auths": {
                    self.auth_server: {
                        "auth": base64.b64encode(
                            f"{self.username}:{self.password}".encode()
                        ).decode(),
                    }
                }
            }


@dataclass
class TargetSpec:
    saas_file: SaasFile
    resource_template: SaasResourceTemplate
    target: SaasResourceTemplateTarget
    image_auth: ImageAuth
    hash_length: int
    github: Github
    target_config_hash: str
    secret_reader: SecretReaderBase

    @property
    def saas_file_name(self) -> str:
        return self.saas_file.name

    @property
    def managed_resource_types(self) -> Iterable[str]:
        return self.saas_file.managed_resource_types

    @property
    def managed_resource_names(self) -> Sequence[ManagedResourceName] | None:
        return self.saas_file.managed_resource_names

    @property
    def privileged(self) -> bool:
        return bool(self.saas_file.cluster_admin)

    @property
    def image_patterns(self) -> list[str]:
        return self.saas_file.image_patterns

    @property
    def resource_template_name(self) -> str:
        return self.resource_template.name

    @property
    def url(self) -> str:
        return self.resource_template.url

    @property
    def path(self) -> str:
        return self.resource_template.path

    @property
    def ref(self) -> str:
        return self.target.ref

    @property
    def provider(self) -> str:
        return self.resource_template.provider or "openshift-template"

    @property
    def cluster(self) -> str:
        return self.target.namespace.cluster.name

    @property
    def namespace(self) -> str:
        return self.target.namespace.name

    @property
    def delete(self) -> bool:
        return bool(self.target.delete)

    @property
    def html_url(self) -> str:
        git_object = "blob" if self.provider == "openshift-template" else "tree"
        return f"{self.url}/{git_object}/{self.ref}{self.path}"

    @property
    def error_prefix(self) -> str:
        return f"[{self.saas_file_name}/{self.resource_template_name}] {self.html_url}:"

    def parameters(self, adjust: bool = True) -> dict[str, Any]:
        environment_parameters = self._collect_parameters(
            self.target.namespace.environment, adjust=adjust
        )
        saas_file_parameters = self._collect_parameters(self.saas_file, adjust=adjust)
        resource_template_parameters = self._collect_parameters(
            self.resource_template, adjust=adjust
        )
        target_parameters = self._collect_parameters(self.target, adjust=adjust)

        try:
            saas_file_secret_parameters = self._collect_secret_parameters(
                self.saas_file
            )
            resource_template_secret_parameters = self._collect_secret_parameters(
                self.resource_template
            )
            environment_secret_parameters = self._collect_secret_parameters(
                self.target.namespace.environment
            )
            target_secret_parameters = self._collect_secret_parameters(self.target)
        except Exception as e:
            logging.error(f"Error collecting secrets: {e}")
            raise

        consolidated_parameters = {}
        consolidated_parameters.update(environment_parameters)
        consolidated_parameters.update(environment_secret_parameters)
        consolidated_parameters.update(saas_file_parameters)
        consolidated_parameters.update(saas_file_secret_parameters)
        consolidated_parameters.update(resource_template_parameters)
        consolidated_parameters.update(resource_template_secret_parameters)
        consolidated_parameters.update(target_parameters)
        consolidated_parameters.update(target_secret_parameters)

        for replace_key, replace_value in consolidated_parameters.items():
            if not isinstance(replace_value, str):
                continue
            replace_pattern = "${" + replace_key + "}"
            for k, v in consolidated_parameters.items():
                if not isinstance(v, str):
                    continue
                if replace_pattern in v:
                    consolidated_parameters[k] = v.replace(
                        replace_pattern, replace_value
                    )

        return consolidated_parameters

    @staticmethod
    def _collect_parameters(
        container: HasParameters, adjust: bool = True
    ) -> dict[str, str]:
        parameters = container.parameters or {}
        if isinstance(parameters, str):
            parameters = json.loads(parameters)
        if adjust:
            # adjust Python's True/False
            for k, v in parameters.items():
                if v is True:
                    parameters[k] = "true"
                elif v is False:
                    parameters[k] = "false"
                elif any(isinstance(v, t) for t in [dict, list, tuple]):
                    parameters[k] = json.dumps(v)
        return parameters

    def _collect_secret_parameters(
        self, container: HasSecretParameters
    ) -> dict[str, str]:
        return {
            sp.name: self.secret_reader.read_secret(sp.secret)
            for sp in container.secret_parameters or []
        }
