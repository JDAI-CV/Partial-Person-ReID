## Multi-domain Results

R34-ibn Backbone / SGD / 15 domain datasets / gempooling
| Method | DukeMTMC unseen| Market1501 seen| MSMT17 seen|
|:--:|:--:|:--:|:--:|
| Baseline  |77.33(61.46) |96.14(89.12)|80.44(55.04)| 
| Baseline + Camera-aware | 80.12(63.90)|96.56(89.43)|81.43(56.33)| 
*****************************************************
R34-ibn Backbone / SGD / 14 domain datasets / gempooling
| Method | DukeMTMC unseen| Market1501 unseen| MSMT17 |
|:--:|:--:|:--:|:--:|
| Baseline  |77.33(60.98) |89.43(72.03)|79.90(54.47)| 
| Baseline + Camera-aware | 79.44(62.63)|90.08(73.52)|81.30(55.99)| 



*****************************************************
SGD / 14 domain datasets + bjz / gempooling
| Backbone |PartialREID unseen| OccludedREID unseen | PartialiLIDS unseen| DukeMTMC unseen| Market1501 unseen| MSMT17 seen | bjzCrowd seen | bjzBlack seen|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| R34_ibn  |83.67(83.55) |92.44(94.59)|93.40(90.16)| 79.71(65.12)| 90.94(75.43)| 82.18(57.60)| 83.56(85.97)| 42.62(43.70)|
| R101_ibn |86.33(82.98)|88.24(92.17)|92.40(89.63)| 80.25(65.12)| 91.36(77.16)| 83.93(61.90)| -| -| 

