# SOP-BL-02: Bleach Tower Temperature Control

## Purpose
How bleach tower temperature is controlled and what to do when it deviates.

## Control loop
Tower temperature (tag bleach_temp) is controlled by direct steam injection through control valve
TV-210. The valve positioner is air-operated. Target is 75 °C; alarm limits 70–80 °C.

## Common failure modes
- Positioner air supply leak or low instrument air pressure: the valve cannot open fully and
  temperature drifts down over 20–60 minutes.
- Steam header pressure drop (boiler trip or high demand elsewhere): all steam users are affected,
  not only the bleach tower.
- Temperature transmitter failure: reading jumps or freezes; verify with a portable thermometer.

## Impact
Low temperature slows the bleaching reaction. Expect about 0.3 %ISO brightness loss per °C below
target, appearing roughly 30 minutes later at the brightness sensor.

## Actions
1. Check valve position versus controller output. A large gap points to the positioner or air supply.
2. Check steam header pressure.
3. Raise a maintenance work order for TV-210 if the valve does not follow its output.
