"""
Component that will perform object detection and identification via CodeProject.AI Server.

For more details about this platform, please refer to the documentation at
https://home-assistant.io/components/image_processing.codeproject_ai_object
"""

from __future__ import annotations

from collections import Counter, namedtuple
from datetime import datetime
import io
import logging
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, UnidentifiedImageError
import voluptuous as vol

import codeprojectai.core as cpai

import homeassistant.helpers.config_validation as cv
import homeassistant.util.dt as dt_util
from homeassistant.components.image_processing import (
    ATTR_CONFIDENCE,
    CONF_CONFIDENCE,
    CONF_ENTITY_ID,
    CONF_NAME,
    CONF_SOURCE,
    DEFAULT_CONFIDENCE,
    PLATFORM_SCHEMA,
    ImageProcessingEntity,
)
from homeassistant.const import ATTR_ENTITY_ID, CONF_IP_ADDRESS, CONF_PORT
from homeassistant.core import HomeAssistant
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util.pil import draw_box

_LOGGER = logging.getLogger(__name__)

ANIMAL = "animal"
ANIMALS = [
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
]
OTHER = "other"
PERSON = "person"
VEHICLE = "vehicle"
VEHICLES = ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck"]

CONF_TARGET = "target"
CONF_TARGETS = "targets"
CONF_TIMEOUT = "timeout"
CONF_SAVE_FILE_FORMAT = "save_file_format"
CONF_SAVE_FILE_FOLDER = "save_file_folder"
CONF_SAVE_TIMESTAMPTED_FILE = "save_timestamped_file"
CONF_ALWAYS_SAVE_LATEST_FILE = "always_save_latest_file"
CONF_USE_SUBFOLDERS = "use_subfolders"
CONF_FILENAME_PREFIX = "filename_prefix"
CONF_OBJECT_BOX_COLOUR = "object_box_colour"
CONF_ROI_BOX_COLOUR = "roi_box_colour"
CONF_SHOW_BOXES = "show_boxes"
CONF_ROI_Y_MIN = "roi_y_min"
CONF_ROI_X_MIN = "roi_x_min"
CONF_ROI_Y_MAX = "roi_y_max"
CONF_ROI_X_MAX = "roi_x_max"
CONF_SCALE = "scale"
CONF_CUSTOM_MODEL = "custom_model"
CONF_CROP_ROI = "crop_to_roi"

DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S-%f"
DEFAULT_TARGETS = [{CONF_TARGET: PERSON}]
DEFAULT_TIMEOUT = 10
DEFAULT_ROI_Y_MIN = 0.0
DEFAULT_ROI_Y_MAX = 1.0
DEFAULT_ROI_X_MIN = 0.0
DEFAULT_ROI_X_MAX = 1.0
DEFAULT_SCALE = 1.0
DEFAULT_ROI = (
    DEFAULT_ROI_Y_MIN,
    DEFAULT_ROI_X_MIN,
    DEFAULT_ROI_Y_MAX,
    DEFAULT_ROI_X_MAX,
)

EVENT_OBJECT_DETECTED = "codeproject_ai.object_detected"
SAVED_FILE = "saved_file"
MIN_CONFIDENCE = 0.1
JPG = "jpg"
PNG = "png"

RED = (255, 0, 0)
GREEN = (0, 255, 0)

TARGETS_SCHEMA = {
    vol.Required(CONF_TARGET): cv.string,
    vol.Optional(CONF_CONFIDENCE): vol.All(
        vol.Coerce(float), vol.Range(min=10, max=100)
    ),
}

PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend(
    {
        vol.Required(CONF_IP_ADDRESS): cv.string,
        vol.Required(CONF_PORT): cv.port,
        vol.Optional(CONF_TIMEOUT, default=DEFAULT_TIMEOUT): cv.positive_int,
        vol.Optional(CONF_CUSTOM_MODEL, default=""): cv.string,
        vol.Optional(CONF_TARGETS, default=DEFAULT_TARGETS): vol.All(
            cv.ensure_list, [vol.Schema(TARGETS_SCHEMA)]
        ),
        vol.Optional(CONF_ROI_Y_MIN, default=DEFAULT_ROI_Y_MIN): cv.small_float,
        vol.Optional(CONF_ROI_X_MIN, default=DEFAULT_ROI_X_MIN): cv.small_float,
        vol.Optional(CONF_ROI_Y_MAX, default=DEFAULT_ROI_Y_MAX): cv.small_float,
        vol.Optional(CONF_ROI_X_MAX, default=DEFAULT_ROI_X_MAX): cv.small_float,
        vol.Optional(CONF_SCALE, default=DEFAULT_SCALE): vol.All(
            vol.Coerce(float), vol.Range(min=0.1, max=1.0)
        ),
        vol.Optional(CONF_SAVE_FILE_FOLDER): cv.string,
        vol.Optional(CONF_SAVE_FILE_FORMAT, default=JPG): vol.In([JPG, PNG]),
        vol.Optional(CONF_SAVE_TIMESTAMPTED_FILE, default=False): cv.boolean,
        vol.Optional(CONF_ALWAYS_SAVE_LATEST_FILE, default=False): cv.boolean,
        vol.Optional(CONF_USE_SUBFOLDERS, default=False): cv.boolean,
        vol.Optional(CONF_FILENAME_PREFIX, default=""): cv.string,
        vol.Optional(CONF_OBJECT_BOX_COLOUR, default="#FF0000"): cv.string,
        vol.Optional(CONF_ROI_BOX_COLOUR, default="#00FF00"): cv.string,
        vol.Optional(CONF_SHOW_BOXES, default=True): cv.boolean,
        vol.Optional(CONF_CROP_ROI, default=False): cv.boolean,
    }
)

Box = namedtuple("Box", "y_min x_min y_max x_max")
Point = namedtuple("Point", "y x")


def point_in_box(box: Box, point: Point) -> bool:
    """Return True if point lies in box."""
    return (box.x_min <= point.x <= box.x_max) and (box.y_min <= point.y <= box.y_max)


def object_in_roi(roi: dict[str, float], centroid: dict[str, float]) -> bool:
    """Convenience to convert dicts to Point and Box."""
    target_center_point = Point(centroid["y"], centroid["x"])
    roi_box = Box(roi["y_min"], roi["x_min"], roi["y_max"], roi["x_max"])
    return point_in_box(roi_box, target_center_point)


def get_valid_filename(name: str) -> str:
    """Return a filesystem-safe filename stem."""
    return re.sub(r"(?u)[^-\w.]", "", str(name).strip().replace(" ", "_"))


def get_object_type(object_name: str) -> str:
    """Return normalized object type."""
    if object_name == PERSON:
        return PERSON
    if object_name in ANIMALS:
        return ANIMAL
    if object_name in VEHICLES:
        return VEHICLE
    return OTHER


def hex_to_rgb(value: str, default: tuple[int, int, int]) -> tuple[int, int, int]:
    """Convert #RRGGBB to RGB tuple."""
    try:
        cleaned = value.lstrip("#")
        if len(cleaned) != 6:
            raise ValueError("Invalid RGB hex length")
        return tuple(int(cleaned[i : i + 2], 16) for i in (0, 2, 4))
    except (ValueError, TypeError):
        _LOGGER.warning("Invalid color '%s', falling back to %s", value, default)
        return default


def get_objects(predictions: list[dict[str, Any]], img_width: int, img_height: int) -> list[dict[str, Any]]:
    """Return objects with formatting and extra info."""
    objects: list[dict[str, Any]] = []
    decimal_places = 3

    for pred in predictions:
        box_width = pred["x_max"] - pred["x_min"]
        box_height = pred["y_max"] - pred["y_min"]
        box = {
            "height": round(box_height / img_height, decimal_places),
            "width": round(box_width / img_width, decimal_places),
            "y_min": round(pred["y_min"] / img_height, decimal_places),
            "x_min": round(pred["x_min"] / img_width, decimal_places),
            "y_max": round(pred["y_max"] / img_height, decimal_places),
            "x_max": round(pred["x_max"] / img_width, decimal_places),
        }
        box_area = round(box["height"] * box["width"], decimal_places)
        centroid = {
            "x": round(box["x_min"] + (box["width"] / 2), decimal_places),
            "y": round(box["y_min"] + (box["height"] / 2), decimal_places),
        }
        name = pred["label"]
        object_type = get_object_type(name)
        confidence = round(pred["confidence"] * 100, decimal_places)

        objects.append(
            {
                "bounding_box": box,
                "box_area": box_area,
                "centroid": centroid,
                "name": name,
                "object_type": object_type,
                "confidence": confidence,
            }
        )

    return objects


def _build_entities(config: ConfigType) -> list["ObjectClassifyEntity"]:
    """Build entities from config."""
    save_file_folder = config.get(CONF_SAVE_FILE_FOLDER)
    use_subfolders = config.get(CONF_USE_SUBFOLDERS, False)

    base_save_path = Path(save_file_folder) if save_file_folder else None
    entities: list[ObjectClassifyEntity] = []

    for camera in config[CONF_SOURCE]:
        camera_entity_id = camera[CONF_ENTITY_ID]
        camera_name = camera.get(CONF_NAME) or camera_entity_id.split(".", 1)[1]

        if base_save_path and use_subfolders:
            camera_save_file_folder = base_save_path / camera_entity_id.split(".", 1)[1]
        else:
            camera_save_file_folder = base_save_path

        entities.append(
            ObjectClassifyEntity(
                ip_address=config[CONF_IP_ADDRESS],
                port=config[CONF_PORT],
                timeout=config[CONF_TIMEOUT],
                custom_model=config[CONF_CUSTOM_MODEL],
                targets=config[CONF_TARGETS],
                confidence=config.get(CONF_CONFIDENCE, DEFAULT_CONFIDENCE),
                roi_y_min=config[CONF_ROI_Y_MIN],
                roi_x_min=config[CONF_ROI_X_MIN],
                roi_y_max=config[CONF_ROI_Y_MAX],
                roi_x_max=config[CONF_ROI_X_MAX],
                scale=config[CONF_SCALE],
                show_boxes=config[CONF_SHOW_BOXES],
                save_file_folder=camera_save_file_folder,
                save_file_format=config[CONF_SAVE_FILE_FORMAT],
                save_timestamped_file=config[CONF_SAVE_TIMESTAMPTED_FILE],
                always_save_latest_file=config[CONF_ALWAYS_SAVE_LATEST_FILE],
                use_subfolders=use_subfolders,
                filename_prefix=config[CONF_FILENAME_PREFIX],
                object_box_colour=config[CONF_OBJECT_BOX_COLOUR],
                roi_box_colour=config[CONF_ROI_BOX_COLOUR],
                crop_roi=config[CONF_CROP_ROI],
                camera_entity=camera[CONF_ENTITY_ID],
                name=camera_name,
            )
        )

    return entities


def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities,
    discovery_info: DiscoveryInfoType | None = None,
) -> None:
    """Set up the platform."""
    add_entities(_build_entities(config))


async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities,
    discovery_info: DiscoveryInfoType | None = None,
) -> None:
    """Set up the platform."""
    async_add_entities(_build_entities(config))


class ObjectClassifyEntity(ImageProcessingEntity):
    """Perform object classification via CodeProject.AI."""

    _attr_should_poll = False

    def __init__(
        self,
        ip_address: str,
        port: int,
        timeout: int,
        custom_model: str,
        targets: list[dict[str, Any]],
        confidence: float,
        roi_y_min: float,
        roi_x_min: float,
        roi_y_max: float,
        roi_x_max: float,
        scale: float,
        show_boxes: bool,
        save_file_folder: Path | None,
        save_file_format: str,
        save_timestamped_file: bool,
        always_save_latest_file: bool,
        use_subfolders: bool,
        filename_prefix: str,
        object_box_colour: str,
        roi_box_colour: str,
        crop_roi: bool,
        camera_entity: str,
        name: str | None = None,
    ) -> None:
        """Initialize the entity."""
        super().__init__()

        self._cpai_object = cpai.CodeProjectAIObject(
            ip=ip_address,
            port=port,
            timeout=timeout,
            min_confidence=MIN_CONFIDENCE,
            custom_model=custom_model,
        )

        self.timeout = timeout
        self._attr_camera_entity = camera_entity
        self._attr_confidence = confidence

        self._custom_model = custom_model
        self._targets = [dict(target) for target in targets]
        for target in self._targets:
            if CONF_CONFIDENCE not in target:
                target[CONF_CONFIDENCE] = confidence

        self._targets_names = [target[CONF_TARGET] for target in self._targets]
        self._object_box_colour = hex_to_rgb(object_box_colour, RED)
        self._roi_box_colour = hex_to_rgb(roi_box_colour, GREEN)
        self._filename_prefix = filename_prefix or ""

        self._camera_name = camera_entity.split(".", 1)[1]
        self._attr_name = name or f"{self._filename_prefix}{self._camera_name}"

        self._state: int = 0
        self._objects: list[dict[str, Any]] = []
        self._targets_found: list[dict[str, Any]] = []
        self._summary: dict[str, int] = {}
        self._last_detection: str | None = None

        self._roi_dict = {
            "y_min": roi_y_min,
            "x_min": roi_x_min,
            "y_max": roi_y_max,
            "x_max": roi_x_max,
        }
        self._crop_roi = crop_roi
        self._scale = scale
        self._show_boxes = show_boxes
        self._image_width: int | None = None
        self._image_height: int | None = None
        self._save_file_folder = save_file_folder
        self._save_file_format = save_file_format
        self._always_save_latest_file = always_save_latest_file
        self._save_timestamped_file = save_timestamped_file
        self._use_subfolders = use_subfolders
        self._image: Image.Image | None = None

    @property
    def state(self) -> int:
        """Return the state of the entity."""
        return self._state

    @property
    def unit_of_measurement(self) -> str:
        """Return unit of measurement."""
        return "targets"

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return device specific state attributes."""
        attr: dict[str, Any] = {}
        attr["targets"] = self._targets
        attr["targets_found"] = [
            {obj["name"]: obj["confidence"]} for obj in self._targets_found
        ]
        attr["summary"] = self._summary
        attr["all_objects"] = [
            {obj["name"]: obj["confidence"]} for obj in self._objects
        ]

        if self._last_detection:
            attr["last_target_detection"] = self._last_detection

        if self._custom_model:
            attr["custom_model"] = self._custom_model

        if self._save_file_folder:
            attr[CONF_SAVE_FILE_FOLDER] = str(self._save_file_folder)
            attr[CONF_SAVE_FILE_FORMAT] = self._save_file_format
            attr[CONF_SAVE_TIMESTAMPTED_FILE] = self._save_timestamped_file
            attr[CONF_ALWAYS_SAVE_LATEST_FILE] = self._always_save_latest_file
            attr[CONF_USE_SUBFOLDERS] = self._use_subfolders
            attr[CONF_FILENAME_PREFIX] = self._filename_prefix

        return attr

    def process_image(self, image: bytes) -> None:
        """Process an image."""
        self._state = 0
        self._objects = []
        self._targets_found = []
        self._summary = {}

        try:
            self._image = Image.open(io.BytesIO(bytearray(image)))
        except UnidentifiedImageError:
            _LOGGER.warning("Unable to open image from camera")
            return

        self._image_width, self._image_height = self._image.size

        if self._crop_roi:
            roi = (
                self._image_width * self._roi_dict["x_min"],
                self._image_height * self._roi_dict["y_min"],
                self._image_width * self._roi_dict["x_max"],
                self._image_height * self._roi_dict["y_max"],
            )
            self._image = self._image.crop(roi)
            self._image_width, self._image_height = self._image.size
            with io.BytesIO() as output:
                self._image.save(output, format="JPEG")
                image = output.getvalue()

        if self._scale != DEFAULT_SCALE:
            newsize = (
                int(self._image_width * self._scale),
                int(self._image_height * self._scale),
            )
            self._image.thumbnail(newsize, Image.LANCZOS)
            self._image_width, self._image_height = self._image.size
            with io.BytesIO() as output:
                self._image.save(output, format="JPEG")
                image = output.getvalue()

        saved_image_path = None

        try:
            predictions = self._cpai_object.detect(image)
        except cpai.CodeProjectAIServerException as exc:
            _LOGGER.error("CodeProject.AI Server error: %s", exc)
            return

        self._objects = get_objects(predictions, self._image_width, self._image_height)

        for obj in self._objects:
            if obj["name"] not in self._targets_names and obj["object_type"] not in self._targets_names:
                continue

            required_confidence = self._attr_confidence or DEFAULT_CONFIDENCE

            for target in self._targets:
                if obj["object_type"] == target[CONF_TARGET]:
                    required_confidence = target[CONF_CONFIDENCE]

            for target in self._targets:
                if obj["name"] == target[CONF_TARGET]:
                    required_confidence = target[CONF_CONFIDENCE]

            if obj["confidence"] >= required_confidence:
                if not self._crop_roi and not object_in_roi(self._roi_dict, obj["centroid"]):
                    continue
                self._targets_found.append(obj)

        self._state = len(self._targets_found)

        if self._state > 0:
            self._last_detection = dt_util.now().strftime(DATETIME_FORMAT)

        targets_found = [obj["name"] for obj in self._targets_found]
        self._summary = dict(Counter(targets_found))

        if self._save_file_folder and (self._state > 0 or self._always_save_latest_file):
            try:
                self._save_file_folder.mkdir(parents=True, exist_ok=True)
                saved_image_path = self.save_image(self._targets_found, self._save_file_folder)
            except OSError as exc:
                _LOGGER.error("Unable to save image to %s: %s", self._save_file_folder, exc)

        for target in self._targets_found:
            event_data = target.copy()
            event_data[ATTR_ENTITY_ID] = self.entity_id
            if saved_image_path:
                event_data[SAVED_FILE] = saved_image_path
            self.hass.bus.fire(EVENT_OBJECT_DETECTED, event_data)

    def save_image(self, targets: list[dict[str, Any]], directory: Path) -> str | None:
        """Draw bounding boxes and save the processed image."""
        if self._image is None:
            return None

        try:
            img = self._image.convert("RGB")
        except UnidentifiedImageError:
            _LOGGER.warning("CodeProject.AI unable to process image, bad data")
            return None

        draw = ImageDraw.Draw(img)

        roi_tuple = tuple(self._roi_dict.values())
        if roi_tuple != DEFAULT_ROI and self._show_boxes and not self._crop_roi:
            draw_box(
                draw,
                roi_tuple,
                img.width,
                img.height,
                text="ROI",
                color=self._roi_box_colour,
            )

        if self._show_boxes:
            for obj in targets:
                box = obj["bounding_box"]
                centroid = obj["centroid"]
                box_label = f'{obj["name"]}: {obj["confidence"]:.1f}%'

                draw_box(
                    draw,
                    (box["y_min"], box["x_min"], box["y_max"], box["x_max"]),
                    img.width,
                    img.height,
                    text=box_label,
                    color=self._object_box_colour,
                )

                draw.text(
                    (centroid["x"] * img.width, centroid["y"] * img.height),
                    text="X",
                    fill=self._object_box_colour,
                )

        latest_filename = (
            f"{get_valid_filename(self._attr_name).lower()}_latest.{self._save_file_format}"
        )
        latest_save_path = directory / latest_filename
        img.save(latest_save_path)
        _LOGGER.info("CodeProject.AI saved file %s", latest_save_path)

        saved_image_path: Path = latest_save_path

        if self._save_timestamped_file and self._last_detection:
            timestamp_filename = (
                f"{get_valid_filename(self._attr_name).lower()}_{self._last_detection}.{self._save_file_format}"
            )
            timestamp_save_path = directory / timestamp_filename
            img.save(timestamp_save_path)
            _LOGGER.info("CodeProject.AI saved file %s", timestamp_save_path)
            saved_image_path = timestamp_save_path

        return str(saved_image_path)
