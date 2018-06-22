import requests
import json
import os
import re
import ipykernel
from nbformat import read, write, NO_CONVERT

# version dependent imports
from six.moves.urllib.parse import urljoin

try:  # Python 3
    from notebook.notebookapp import list_running_servers
except ImportError:  # Python 2
    import warnings
    from IPython.utils.shimmodule import ShimWarning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ShimWarning)
        from IPython.html.notebookapp import list_running_servers
import six
import errno


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def get_notebook_name():
    """
    Return the full path of the jupyter notebook.
    """
    kernel_id = re.search(
        "kernel-(.*).json", ipykernel.connect.get_connection_file()
    ).group(1)
    servers = list_running_servers()
    for ss in servers:
        response = requests.get(
            urljoin(ss["url"], "api/sessions"), params={"token": ss.get("token", "")}
        )
        for nn in json.loads(response.text):
            if nn["kernel"]["id"] == kernel_id:
                relative_path = nn["notebook"]["path"]
                return os.path.join(ss["notebook_dir"], relative_path)


def get_current_notebook():
    """
    Return the cleaned content and meta data for current notebook
    """
    notebook = read(get_notebook_name(), NO_CONVERT)

    return {
        "path": "/".join(get_notebook_name().split("/")[3:]),
        "content": strip_output(notebook),
    }


def nb_cells(nb):
    """Yield all cells in an nbformat-insensitive manner"""
    if nb.nbformat < 4:
        for ws in nb.worksheets:
            for cell in ws.cells:
                yield cell
    else:
        for cell in nb.cells:
            yield cell


def strip_output(nb):
    """
    Strip the outputs, execution count/prompt number and miscellaneous
    metadata from a notebook object.
    """

    nb.metadata.pop("signature", None)
    nb.metadata.pop("widgets", None)

    for cell in nb_cells(nb):
        # Remove the outputs
        if "outputs" in cell:
            cell["outputs"] = []

        # Remove the prompt_number/execution_count
        if "prompt_number" in cell:
            cell["prompt_number"] = None
        if "execution_count" in cell:
            cell["execution_count"] = None

        # Remove metadata
        for output_style in ["collapsed", "scrolled"]:
            if output_style in cell.metadata:
                cell.metadata[output_style] = False
        if "metadata" in cell:
            for field in ["collapsed", "scrolled", "ExecuteTime"]:
                cell.metadata.pop(field, None)
    return nb


def retry_post(urls, **kwargs):
    """ Retry post request on a list of urls. """
    url = urls.pop()
    try:
        request = requests.post(url, **kwargs)
    except Exception as e:
        if len(urls) == 0:
            raise e
        else:
            request = retry_post(urls, **kwargs)

    if request.status_code != 200:
        raise Exception(
            "Request failed to run by returning code of {}. {}".format(
                request.status_code, request.text
            )
        )
    return request


def post_to_platform(json):
    """ Post graphql query to platform. """
    urls = list(
        map(lambda url: url + "/graphql", os.environ["PLATFORM_URLS"].split(","))
    )

    return retry_post(
        urls,
        cookies={
            "datascience-platform": os.environ["DS_JWT"],
            "_csrf": os.environ["USER_TOKEN"],
        },
        headers={"csrf-token": os.environ["USER_TOKEN"]},
        json=json,
    )
