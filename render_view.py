import argparse
import cv2

from cameras import cam_resize, load_extr_and_intr_camera
from config import load_config
from renderer import make_renderer_with_trained_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--cameras_path", type=str)
    parser.add_argument("--height", nargs="?", type=int, default=None)
    parser.add_argument("--width", nargs="?", type=int, default=None)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    config = load_config(args.config_path)

    renderer = make_renderer_with_trained_model(config)
    def render_view(camCv2world, cam_intrinsic):
        renderer.set_height(cam_intrinsic["height"])
        renderer.set_width(cam_intrinsic["width"])
        return renderer.render(camCv2world, cam_intrinsic["K"])

    camCv2world, K = load_extr_and_intr_camera(args.cameras_path)
    cam_intrinsic = {
        "K": K,
        "height": config["data"]["img_height"],
        "width": config["data"]["img_width"],
    }

    height = args.height if args.height is not None else config["data"]["img_height"]
    width = args.width if args.width is not None else config["data"]["img_width"]
    view = render_view(camCv2world, cam_resize(cam_intrinsic, height, width))

    cv2.imwrite(args.output_path, view[..., ::-1])


if __name__ == "__main__":
    main()
