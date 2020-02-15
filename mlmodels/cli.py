import click
import os
from mlmodels.model_service.build_docker_image import build_model_service_docker_image


@click.group()
def cli():
    pass


@click.command()
@click.argument("s3_model_uri")
@click.argument("tag")
@click.argument("model_version")
def dockerize(s3_model_uri, tag, model_version):
    build_model_service_docker_image(
        s3_model_uri,
        tag,
        model_version,
    )


cli.add_command(dockerize)


if __name__ == '__main__':
    cli()