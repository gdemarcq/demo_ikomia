import streamlit as st
import numpy as np
import base64
import requests
import json
import io
from PIL import Image
from io import BytesIO
from yarl import URL
import logging
import time


logger = logging.getLogger(__name__)


_MAX_SIZE = 2048

if "run_workflow" not in st.session_state:
    st.session_state.run_workflow = True

if "result_img" not in st.session_state:
    st.session_state.result_img = None

if "jwt" not in st.session_state:
    st.session_state.jwt = None

# Ikomia Scale URL
IKSCALE_URL=URL("https://scale.ikomia.ai")

# Ikomia Scale project deployment endpoint
ENDPOINT_URL = ""

# Ikomia Workflow name
TASK_NAME = ""


class HTTPBadCodeError(Exception):
    """Raised when request status code <200 or >299."""

    def __init__(self, url: URL, code: int, content=None):
        """
        Init a new HTTP error.

        Args:
            url: URL
            code: HTTP code
            headers: Response header
            content: Response content
        """
        super().__init__(f"Bad return code {code} on '{url}'")
        self.url = url
        self.code = code
        self.content = content


def request(session, headers, method, url, params=None, data=None):

    if data is not None:
        data = json.dumps(data)

    request = requests.Request(
        method=method,
        url=url,
        headers=headers,
        params=params,
        data=data,
    )
    prepared_request = session.prepare_request(request)

    # Produce some debug logs
    logger.debug("Will %s '%s'", prepared_request.method, prepared_request.url)
    if prepared_request.headers:
        logger.debug(" with headers : %s", prepared_request.headers)
    if prepared_request.body:
        if len(prepared_request.body) > 10240:
            logger.debug(" with body    : .... too long to be dumped ! ...")
        else:
            logger.debug(" with body    : %s", prepared_request.body)

    try:
        response = session.send(
            prepared_request,
            allow_redirects=False,
            timeout=(30, 30),
        )
    except requests.exceptions.ReadTimeout as e:
        return None

    logger.debug("Response code : %d", response.status_code)
    logger.debug(" with headers : %s", response.headers)
    if ("Content-Length" in response.headers and int(response.headers["Content-Length"]) > 10240) or len(
        response.content
    ) > 10240:
        logger.debug(" with content    : .... too long to be dumped ! ...")
    else:
        logger.debug(" with content    : %s", response.content)

    if response.status_code >= 200 and response.status_code <= 299:
        try:
            return response.json()
        except json.JSONDecodeError:
            return response.content

    raise HTTPBadCodeError(url, response.status_code, response.content)


def on_token_or_project_change():
     st.session_state.jwt = None



def pil_image_to_b64str(im):
    buffered = BytesIO()
    im.save(buffered, format="JPEG")
    img_bytes = base64.b64encode(buffered.getvalue())
    img_str = img_bytes.decode('utf-8')
    return img_str


def b64str_to_numpy(base64str):
    base64_img_bytes = base64str.encode('utf-8')
    base64bytes = base64.b64decode(base64_img_bytes)
    bytes_obj = io.BytesIO(base64bytes)
    img = Image.open(bytes_obj)
    return np.asarray(img)


def bytesio_obj_to_b64str(bytes_obj):
    return base64.b64encode(bytes_obj.read()).decode("utf-8")


def bytesio_to_pil_image(bytes_obj):
    data = io.BytesIO(bytes_obj.read())
    img = Image.open(data)
    return img


def numpy_to_pil_image(np_img):
    if np_img.dtype in ["float32", "float64"]:
        return Image.fromarray((np_img * 255 / np.max(np_img)).astype('uint8'))
    else:
        return Image.fromarray(np_img)


def check_image_size(img):
    max_size = max(img.size)
    if max_size > _MAX_SIZE:
        if img.width > img.height:
            img = img.resize((_MAX_SIZE, int(img.height * _MAX_SIZE / img.width)))
        else:
            img = img.resize((int(img.width * _MAX_SIZE / img.height), _MAX_SIZE))

    return img


def make_payload(task_name, task_output_index, img_base64):
    headers = {"Content-Type": "application/json"}

    body = {
        "inputs": [{"image": img_base64}],
        "outputs": [
            {"task_name": task_name, "task_index": 0, "output_index": task_output_index}
        ],
        "parameters": [],
        "isBase64Encoded": True
    }

    return body


def colorize_rows(s):
    return ['background-color: #c5b4e3'] * len(s) if s.name == 'Left' else ['background-color: #add8e6'] * len(s)


def display_result(results_list, img_list, image_placeholder, image):

    for img, res in zip(img_list, results_list):
        if img is not None:
            image_placeholder.empty()
            col1, col2 = st.columns([1, 1])
            col1.image(np.asarray(img), use_column_width="auto", clamp=True, channels="BGR")
            col2.json(res)
        else:
            image_placeholder.empty()
            col1, col2 = st.columns([1, 1])
            col1.image(np.asarray(image), width=512, clamp=True)
            col2.json(res)


def do(api_url, jwt, task_name, task_output_index, image):
    url = URL(api_url)

    session = requests.Session()
    headers = {
        "User-Agent": "IkomiaCli",
        "Authorization": f"Bearer {jwt}",
    }

    # Run workflow
    response = request(session, headers, "PUT", url / "api/run", data=make_payload(task_name, task_output_index, image))

    # Get results
    uuid = response

    response = None

    while response is None or len(response) == 0:
        time.sleep(1)
        response = request(session, headers, "GET", url / f"api/results/{uuid}")

    return response


def process_image(url, jwt, task_name, task_output_index, image_placeholder, image):
    img_base64 = pil_image_to_b64str(image)
    json_data = do(url, jwt, task_name, task_output_index, img_base64)
    img_list = []
    json_list = []
    data_dict = None

    for line in json_data:
        (t, d) = next(iter(line.items()))
        if t == "image":
            output_obj_img = d
            output_img = b64str_to_numpy(output_obj_img)
            output_img = np.array(output_img[:, :, ::-1], dtype='uint8')
            img_list.append(output_img)
            json_list.append({})
        elif t == "DATA_DICT":
            data_dict = d
        else:
            raise ValueError(f"Can't parse {t} in response")


    if data_dict is not None:
        if type(data_dict) is dict:
            # Get base 64 encoded image and its extracted information
            if "crop" in data_dict:
                output_img = b64str_to_numpy(data_dict["crop"])
                output_img = np.array(output_img[:, :, ::-1], dtype='uint8')
                img_list.append(output_img)
                del data_dict["crop"]
            else:
                img_list.append(None)
            json_list.append(data_dict)
        else:
            # data_dict is a list, loop over elements
            for elt in data_dict:
                # Get base 64 encoded image and its extracted information
                if "crop" in elt:
                    output_img = b64str_to_numpy(elt["crop"])
                    output_img = np.array(output_img[:, :, ::-1], dtype='uint8')
                    img_list.append(output_img)
                    del elt["crop"]
                else:
                    img_list.append(None)
                json_list.append(elt)


    # Update image in streamlit view
    display_result(json_list, img_list, image_placeholder, image)


def get_jwt(api_token, endpoint_url):

    if st.session_state.jwt is None:
        session = requests.Session()
        headers = {
            "User-Agent": "IkomiaStreamlit",
            "Authorization": f"Token {api_token}",
        }

        response = request(session, headers, "GET", IKSCALE_URL / f"v1/projects/jwt/", params={"endpoint": endpoint_url})

        if "id_token" in response:
            st.session_state.jwt = response["id_token"]
        else:
            raise Exception("Can't parse response {response}")

    return st.session_state.jwt


def demo():

    st.set_page_config(
        page_title="Ikomia Demo",
        page_icon="./images/ikomia_logo_orange.png",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://www.ikomia.ai",
            "About": f"This app shows how to manage Ikomia Endpoints.",
        },
    )

    #
    #   Sidebar
    #
    st.sidebar.image("./images/ikomia_logo_classic.png", width=200)

    api_token = st.sidebar.text_input("API Token", on_change=on_token_or_project_change)
    with st.sidebar.expander("API", expanded=False):
        endpoint_url = st.text_input("API Endpoint URL", value=ENDPOINT_URL)
        task_name = st.text_input("Ikomia Task Name", value=TASK_NAME)
        task_output_index = st.number_input("Task output index", min_value=0, value=0)

    # Title
    st.title("Ikomia Demo")
    uploaded_input = st.file_uploader("Choose input image")
    image_placeholder = st.empty()
    # Display input image
    if uploaded_input is not None:
        input_img = bytesio_to_pil_image(uploaded_input)
        input_img = check_image_size(input_img)
        image_placeholder.image(np.asarray(input_img), width=512, clamp=True)
    else:
        return

    # Get JWT from ikscale
    jwt = get_jwt(api_token, endpoint_url)

    # Deployment Endpoint invocation
    with st.spinner("Wait for results..."):
        try:
            process_image(endpoint_url, jwt, task_name, task_output_index, image_placeholder, input_img)
        except HTTPBadCodeError as e:
            if e.code == 307:  # Temporary redirect to log in
                st.session_state.jwt = None  # Purge stored JWT
            raise



if __name__ == '__main__':
    demo()
