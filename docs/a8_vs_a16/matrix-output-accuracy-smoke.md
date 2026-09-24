# AQP8 vs AQP16 matrix-output accuracy

`error_A` is FP16 vs AQP8 W4/KV4; `error_B` is FP16 vs AQP16 W4/KV4.
Positive `A−B` or ratio above 1 means AQP8 has the additional error.

| Model / group | error_A (FP16 vs AQP8) | error_B (FP16 vs AQP16) | A−B | A/B |
| --- | ---: | ---: | ---: | ---: |
| llama3.2-3b / Linear | MAE=0.157467<br>RMSE=0.246465<br>relL2=0.384566<br>maxAbs=45.125<br>cos=0.926783 | MAE=0.161436<br>RMSE=0.252279<br>relL2=0.393638<br>maxAbs=37.875<br>cos=0.923428 | MAE=-0.003969<br>RMSE=-0.00581411<br>relL2=-0.0090719<br>maxAbs=+7.25<br>cos=+0.0033546 | MAE=0.975414<br>RMSE=0.976954<br>relL2=0.976954<br>maxAbs=1.19142 |
| llama3.2-3b / QK | MAE=7.12703<br>RMSE=9.73768<br>relL2=0.265421<br>maxAbs=107.125<br>cos=0.96427 | MAE=7.52301<br>RMSE=10.1958<br>relL2=0.277909<br>maxAbs=112.875<br>cos=0.96109 | MAE=-0.395975<br>RMSE=-0.458139<br>relL2=-0.0124876<br>maxAbs=-5.75<br>cos=+0.00318003 | MAE=0.947365<br>RMSE=0.955066<br>relL2=0.955066<br>maxAbs=0.949059 |
| llama3.2-3b / PV | MAE=0.0986252<br>RMSE=0.149356<br>relL2=0.736075<br>maxAbs=1.49438<br>cos=0.793188 | MAE=0.0982712<br>RMSE=0.148245<br>relL2=0.730597<br>maxAbs=1.39307<br>cos=0.795107 | MAE=+0.00035399<br>RMSE=+0.00111168<br>relL2=+0.00547873<br>maxAbs=+0.101318<br>cos=-0.00191897 | MAE=1.0036<br>RMSE=1.0075<br>relL2=1.0075<br>maxAbs=1.07273 |
