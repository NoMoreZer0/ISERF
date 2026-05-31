"""
test_max30102.py -- Level 1 test for MAX30102 pulse sensor on Raspberry Pi 5.

Reads raw IR and red LED samples from the sensor over I2C and prints them.
With no finger on the sensor, IR values should be low (a few hundred).
With a finger pressed gently on the sensor window, IR values should jump
to tens of thousands and oscillate slowly with the user's pulse.

Run with:
    python3 test_max30102.py

Press Ctrl+C to stop.
"""

import time
import smbus2

# MAX30102 I2C configuration
I2C_BUS = 1
I2C_ADDR = 0x57

# MAX30102 register addresses (from the datasheet)
REG_INTR_STATUS_1 = 0x00
REG_FIFO_WR_PTR = 0x04
REG_FIFO_RD_PTR = 0x06
REG_FIFO_DATA = 0x07
REG_FIFO_CONFIG = 0x08
REG_MODE_CONFIG = 0x09
REG_SPO2_CONFIG = 0x0A
REG_LED1_PA = 0x0C  # Red LED current
REG_LED2_PA = 0x0D  # IR LED current
REG_PART_ID = 0xFF


def setup_sensor(bus):
    """Configure MAX30102 for SpO2 mode at 100 Hz sample rate."""
    # Reset the sensor
    bus.write_byte_data(I2C_ADDR, REG_MODE_CONFIG, 0x40)
    time.sleep(0.1)

    # Verify part ID -- should be 0x15 for MAX30102
    part_id = bus.read_byte_data(I2C_ADDR, REG_PART_ID)
    print("Part ID: 0x{:02X} (expected 0x15)".format(part_id))

    # FIFO config: sample averaging = 4, FIFO rollover enabled, almost-full = 17
    bus.write_byte_data(I2C_ADDR, REG_FIFO_CONFIG, 0x4F)

    # Mode config: SpO2 mode (uses both red and IR LEDs)
    bus.write_byte_data(I2C_ADDR, REG_MODE_CONFIG, 0x03)

    # SpO2 config: ADC range 4096nA, sample rate 100 Hz, pulse width 411us (18-bit)
    bus.write_byte_data(I2C_ADDR, REG_SPO2_CONFIG, 0x27)

    # LED current: ~7 mA for red and IR (0x24 = 36 * 0.2 mA)
    bus.write_byte_data(I2C_ADDR, REG_LED1_PA, 0x24)
    bus.write_byte_data(I2C_ADDR, REG_LED2_PA, 0x24)

    # Clear FIFO pointers
    bus.write_byte_data(I2C_ADDR, REG_FIFO_WR_PTR, 0x00)
    bus.write_byte_data(I2C_ADDR, REG_FIFO_RD_PTR, 0x00)


def read_sample(bus):
    """Read one (red, ir) sample from the FIFO. Returns None if no data."""
    wr = bus.read_byte_data(I2C_ADDR, REG_FIFO_WR_PTR)
    rd = bus.read_byte_data(I2C_ADDR, REG_FIFO_RD_PTR)
    num_samples = (wr - rd) & 0x1F
    if num_samples == 0:
        return None

    # Each sample is 6 bytes: 3 for red, 3 for IR (18-bit values)
    data = bus.read_i2c_block_data(I2C_ADDR, REG_FIFO_DATA, 6)
    red = ((data[0] << 16) | (data[1] << 8) | data[2]) & 0x3FFFF
    ir = ((data[3] << 16) | (data[4] << 8) | data[5]) & 0x3FFFF
    return red, ir


def main():
    bus = smbus2.SMBus(I2C_BUS)
    print("Setting up MAX30102...")
    setup_sensor(bus)
    print("Reading samples. Place finger on sensor window. Press Ctrl+C to stop.")
    print()
    print("{:>8s}  {:>8s}  {:>8s}".format("time", "red", "ir"))
    print("{:>8s}  {:>8s}  {:>8s}".format("(s)", "(raw)", "(raw)"))

    t_start = time.time()
    try:
        while True:
            sample = read_sample(bus)
            if sample is not None:
                red, ir = sample
                t = time.time() - t_start
                print("{:8.2f}  {:8d}  {:8d}".format(t, red, ir))
            time.sleep(0.05)  # 20 Hz print rate; sensor still samples at 100 Hz
    except KeyboardInterrupt:
        print()
        print("Stopped.")
    finally:
        bus.close()


if __name__ == "__main__":
    main()
