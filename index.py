"""Yandex Cloud Functions entry point."""

from pharmacheck.cloud_runtime import handle_cloud_function


def handler(event, context):
    return handle_cloud_function(event, context)
