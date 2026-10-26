# SOP-BL-01: Hydrogen Peroxide Bleaching Stage

## Purpose
Defines normal operation of the peroxide (P) bleaching stage and the first checks to make when
brightness after bleaching drops.

## Normal operating window
| Variable | Tag | Target | Alarm limits |
|---|---|---|---|
| H2O2 dose (pump P-301) | peroxide_dose | 25 kg/t | 22–28 kg/t |
| NaOH dose | caustic_dose | 15 kg/t | 13–17 kg/t |
| Tower temperature | bleach_temp | 75 °C | 70–80 °C |
| Tower pH | bleach_ph | 10.8 | 10.4–11.2 |
| Stock consistency | stock_consistency | 12 % | 11–13 % |

Brightness after bleaching (brightness_online) should be 85.5–87 %ISO. Product spec is ≥ 84 %ISO.

## Process relationships
- Brightness responds to peroxide dose with a delay equal to the tower retention time, about
  30 minutes. Rule of thumb: each 1 kg/t of H2O2 is worth roughly 0.3–0.4 %ISO brightness.
- Brightness also falls when tower temperature is low: roughly 0.3 %ISO per °C below 75 °C.
- Very high caustic (> 17 kg/t) causes alkali darkening (yellowing).

## Troubleshooting low brightness
1. Check peroxide_dose against target. If low or unstable, inspect dosing pump P-301
   (diaphragm, check valves, suction strainer) and the H2O2 day tank level.
2. Check bleach_temp. If low, see SOP-BL-02 (steam valve TV-210).
3. Check bleach_ph, but confirm any pH deviation with a grab sample first (SOP-INS-01).
4. If all process variables are normal, suspect the pulp itself: check for a lot or
   supplier change (SOP-QA-02).
5. Remember the 30-minute delay: compare brightness with what the dose was half an hour earlier.
