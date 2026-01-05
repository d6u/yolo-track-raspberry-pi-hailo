# Cat Monitor

## Setup

1. Install Hailo 8 AI HAT on Raspberry Pi 5

2. Install Raspberry Pi OS 12 Debian Bookworm
    - Debian Bookworm comes with Python 3.11, which can come in handy because hailo_platform package might requires older packages that won't be compatible with newer Python.

3. After booting up Pi, install HailoRT PCEI driver and software suite
    - By the time of creating this repo, 4.23 is the newest version for Hailo 8/8L, [PCIE driver](dependencies/hailort-pcie-driver_4.23.0_all.deb) and [software suite](dependencies/hailort_4.23.0_arm64.deb) are included in the `dependencies` folder of this repo.
    - Or you can download fresh copies at https://hailo.ai/developer-zone/software-downloads/?product=ai_accelerators&device=hailo_8_8l.
    - Install steps:

        ```sh
        # Required by PCIE driver
        sudo apt install dkms

        sudo dpkg --install dependencies/hailort-pcie-driver_4.23.0_all.deb dependencies/hailort_4.23.0_arm64.deb
        ```

    - Verify:

        ```sh
        hailortcli fw-control identify
        ```

        Should return something like:

        ```txt
        Executing on device: 0001:01:00.0
        Identifying board
        Control Protocol Version: 2
        Firmware Version: 4.23.0 (release,app,extended context switch buffer)
        Logger Version: 0
        Board Name: Hailo-8
        Device Architecture: HAILO8
        ```

4. Install system python packages
    ```sh
    sudo apt install python3-opencv
    ```
    We cannot install opencv-python package at local project level, otherwise it won't be able to find GStreamer for video writer.

5. Install uv, then use uv to setup project:

    ```sh
    uv venv --system-site-packages
    uv sync
    ```

## Running

```sh
uv run python main.py --help
```

Example usage:

```sh
uv run python main.py --track --duration 30
```

Record a 30s video output at `recordings` directory.
