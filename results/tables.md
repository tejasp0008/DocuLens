# DocuLens result tables

Split(s) evaluated: `test`. Rows: 1440. Documents: 18.

## Mean CER by degradation and method

| degradation | baseline_raw | pipeline_basic | pipeline_full | pipeline_full_otsu | pipeline_clahe |
| --- | --- | --- | --- | --- | --- |
| blur | 0.0002 | 0.0129 | 0.0169 | 0.0345 | 0.0132 |
| jpeg | 0.0003 | 0.0034 | 0.0052 | 0.0195 | 0.0029 |
| lowlight | 0.3688 | 0.3736 | 0.0087 | 0.4235 | 0.0089 |
| noise | 0.0119 | 0.0039 | 0.0080 | 0.0203 | 0.0163 |
| none | 0.0003 | 0.0020 | 0.0084 | 0.0232 | 0.0077 |
| skew | 0.0183 | 0.0304 | 0.0059 | 0.0191 | 0.0039 |

## Overall, per method

| method | cer_mean | cer_median | wer_mean | runtime_ms_mean | preprocess_ms_mean | n |
| --- | --- | --- | --- | --- | --- | --- |
| pipeline_full | 0.0089 | 0.0073 | 0.0898 | 4073.5622 | 2700.5244 | 288 |
| pipeline_clahe | 0.0090 | 0.0049 | 0.0765 | 3841.3585 | 2460.5410 | 288 |
| baseline_raw | 0.0749 | 0.0006 | 0.1082 | 1379.2450 | 0.0005 | 288 |
| pipeline_basic | 0.0796 | 0.0049 | 0.1439 | 1341.4056 | 3.8836 | 288 |
| pipeline_full_otsu | 0.0984 | 0.0217 | 0.2567 | 3651.1757 | 2336.6235 | 288 |

## By degradation

| degradation | method | cer_mean | cer_median | cer_std | wer_mean | runtime_ms_mean | n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| blur | baseline_raw | 0.0002 | 0.0000 | 0.0005 | 0.0018 | 1442.0844 | 54 |
| blur | pipeline_basic | 0.0129 | 0.0135 | 0.0092 | 0.0997 | 1444.9519 | 54 |
| blur | pipeline_clahe | 0.0132 | 0.0088 | 0.0118 | 0.0953 | 3736.5162 | 54 |
| blur | pipeline_full | 0.0169 | 0.0157 | 0.0100 | 0.1335 | 4161.7619 | 54 |
| blur | pipeline_full_otsu | 0.0345 | 0.0255 | 0.0245 | 0.2260 | 3585.4546 | 54 |
| jpeg | baseline_raw | 0.0003 | 0.0000 | 0.0005 | 0.0020 | 1446.4602 | 54 |
| jpeg | pipeline_basic | 0.0034 | 0.0029 | 0.0023 | 0.0309 | 1428.3960 | 54 |
| jpeg | pipeline_clahe | 0.0029 | 0.0024 | 0.0023 | 0.0288 | 4069.1132 | 54 |
| jpeg | pipeline_full | 0.0052 | 0.0049 | 0.0032 | 0.0603 | 4552.1194 | 54 |
| jpeg | pipeline_full_otsu | 0.0195 | 0.0195 | 0.0074 | 0.1743 | 3658.3046 | 54 |
| lowlight | baseline_raw | 0.3688 | 0.5262 | 0.2640 | 0.4429 | 976.3804 | 54 |
| lowlight | pipeline_basic | 0.3736 | 0.5278 | 0.2666 | 0.4512 | 1058.1849 | 54 |
| lowlight | pipeline_clahe | 0.0089 | 0.0098 | 0.0041 | 0.1065 | 3741.2008 | 54 |
| lowlight | pipeline_full | 0.0087 | 0.0097 | 0.0035 | 0.1066 | 4287.4289 | 54 |
| lowlight | pipeline_full_otsu | 0.4235 | 0.5586 | 0.2937 | 0.5697 | 3160.9825 | 54 |
| noise | baseline_raw | 0.0119 | 0.0012 | 0.0171 | 0.0474 | 1681.2114 | 54 |
| noise | pipeline_basic | 0.0039 | 0.0035 | 0.0027 | 0.0300 | 1438.4892 | 54 |
| noise | pipeline_clahe | 0.0163 | 0.0109 | 0.0144 | 0.1131 | 3702.0869 | 54 |
| noise | pipeline_full | 0.0080 | 0.0085 | 0.0040 | 0.0921 | 3783.7986 | 54 |
| noise | pipeline_full_otsu | 0.0203 | 0.0207 | 0.0054 | 0.1764 | 3804.1213 | 54 |
| none | baseline_raw | 0.0003 | 0.0000 | 0.0005 | 0.0025 | 1438.5064 | 18 |
| none | pipeline_basic | 0.0020 | 0.0018 | 0.0016 | 0.0161 | 1418.3244 | 18 |
| none | pipeline_clahe | 0.0077 | 0.0079 | 0.0033 | 0.0910 | 3647.2077 | 18 |
| none | pipeline_full | 0.0084 | 0.0085 | 0.0038 | 0.0988 | 3813.2619 | 18 |
| none | pipeline_full_otsu | 0.0232 | 0.0272 | 0.0097 | 0.2139 | 3961.7236 | 18 |
| skew | baseline_raw | 0.0183 | 0.0100 | 0.0193 | 0.0819 | 1330.3345 | 54 |
| skew | pipeline_basic | 0.0304 | 0.0191 | 0.0263 | 0.1503 | 1311.3665 | 54 |
| skew | pipeline_clahe | 0.0039 | 0.0035 | 0.0024 | 0.0338 | 4022.5922 | 54 |
| skew | pipeline_full | 0.0059 | 0.0059 | 0.0024 | 0.0532 | 3669.4691 | 54 |
| skew | pipeline_full_otsu | 0.0191 | 0.0194 | 0.0047 | 0.1512 | 3943.4993 | 54 |

## By degradation and severity level

| degradation | level | method | cer_mean | wer_mean | runtime_ms_mean | n |
| --- | --- | --- | --- | --- | --- | --- |
| blur | 1 | baseline_raw | 0.0002 | 0.0015 | 1436.5203 | 18 |
| blur | 1 | pipeline_basic | 0.0021 | 0.0180 | 1483.9460 | 18 |
| blur | 1 | pipeline_clahe | 0.0025 | 0.0272 | 3693.8200 | 18 |
| blur | 1 | pipeline_full | 0.0066 | 0.0754 | 4632.9113 | 18 |
| blur | 1 | pipeline_full_otsu | 0.0143 | 0.1238 | 3773.4512 | 18 |
| blur | 2 | baseline_raw | 0.0002 | 0.0015 | 1408.1743 | 18 |
| blur | 2 | pipeline_basic | 0.0147 | 0.1374 | 1442.7234 | 18 |
| blur | 2 | pipeline_clahe | 0.0084 | 0.0532 | 3327.3566 | 18 |
| blur | 2 | pipeline_full | 0.0163 | 0.1459 | 3733.3874 | 18 |
| blur | 2 | pipeline_full_otsu | 0.0241 | 0.2067 | 3243.1763 | 18 |
| blur | 3 | baseline_raw | 0.0003 | 0.0025 | 1481.5586 | 18 |
| blur | 3 | pipeline_basic | 0.0220 | 0.1438 | 1408.1864 | 18 |
| blur | 3 | pipeline_clahe | 0.0286 | 0.2056 | 4188.3719 | 18 |
| blur | 3 | pipeline_full | 0.0277 | 0.1793 | 4118.9869 | 18 |
| blur | 3 | pipeline_full_otsu | 0.0652 | 0.3474 | 3739.7363 | 18 |
| jpeg | 1 | baseline_raw | 0.0003 | 0.0020 | 1496.2031 | 18 |
| jpeg | 1 | pipeline_basic | 0.0029 | 0.0224 | 1431.1453 | 18 |
| jpeg | 1 | pipeline_clahe | 0.0046 | 0.0459 | 4123.4783 | 18 |
| jpeg | 1 | pipeline_full | 0.0075 | 0.0872 | 4299.0409 | 18 |
| jpeg | 1 | pipeline_full_otsu | 0.0256 | 0.2227 | 2791.2147 | 18 |
| jpeg | 2 | baseline_raw | 0.0003 | 0.0020 | 1435.9074 | 18 |
| jpeg | 2 | pipeline_basic | 0.0035 | 0.0337 | 1436.2506 | 18 |
| jpeg | 2 | pipeline_clahe | 0.0022 | 0.0244 | 3405.3301 | 18 |
| jpeg | 2 | pipeline_full | 0.0056 | 0.0705 | 4532.5854 | 18 |
| jpeg | 2 | pipeline_full_otsu | 0.0152 | 0.1434 | 3885.1962 | 18 |
| jpeg | 3 | baseline_raw | 0.0003 | 0.0020 | 1407.2700 | 18 |
| jpeg | 3 | pipeline_basic | 0.0037 | 0.0365 | 1417.7921 | 18 |
| jpeg | 3 | pipeline_clahe | 0.0019 | 0.0161 | 4678.5313 | 18 |
| jpeg | 3 | pipeline_full | 0.0025 | 0.0233 | 4824.7319 | 18 |
| jpeg | 3 | pipeline_full_otsu | 0.0176 | 0.1569 | 4298.5030 | 18 |
| lowlight | 1 | baseline_raw | 0.0005 | 0.0044 | 1340.0013 | 18 |
| lowlight | 1 | pipeline_basic | 0.0021 | 0.0171 | 1507.1771 | 18 |
| lowlight | 1 | pipeline_clahe | 0.0087 | 0.1048 | 3920.9675 | 18 |
| lowlight | 1 | pipeline_full | 0.0087 | 0.1066 | 4799.4809 | 18 |
| lowlight | 1 | pipeline_full_otsu | 0.0197 | 0.1823 | 3999.9089 | 18 |
| lowlight | 2 | baseline_raw | 0.5255 | 0.5530 | 823.7091 | 18 |
| lowlight | 2 | pipeline_basic | 0.5278 | 0.5617 | 848.9791 | 18 |
| lowlight | 2 | pipeline_clahe | 0.0088 | 0.1028 | 3370.9903 | 18 |
| lowlight | 2 | pipeline_full | 0.0087 | 0.1066 | 4268.4244 | 18 |
| lowlight | 2 | pipeline_full_otsu | 0.5597 | 0.6866 | 2772.8720 | 18 |
| lowlight | 3 | baseline_raw | 0.5805 | 0.7713 | 765.4309 | 18 |
| lowlight | 3 | pipeline_basic | 0.5909 | 0.7748 | 818.3983 | 18 |
| lowlight | 3 | pipeline_clahe | 0.0093 | 0.1119 | 3931.6445 | 18 |
| lowlight | 3 | pipeline_full | 0.0087 | 0.1066 | 3794.3813 | 18 |
| lowlight | 3 | pipeline_full_otsu | 0.6912 | 0.8403 | 2710.1667 | 18 |
| noise | 1 | baseline_raw | 0.0003 | 0.0025 | 1670.3281 | 18 |
| noise | 1 | pipeline_basic | 0.0027 | 0.0214 | 1407.6263 | 18 |
| noise | 1 | pipeline_clahe | 0.0082 | 0.0971 | 3657.0695 | 18 |
| noise | 1 | pipeline_full | 0.0083 | 0.0980 | 3647.8335 | 18 |
| noise | 1 | pipeline_full_otsu | 0.0212 | 0.1890 | 3988.6632 | 18 |
| noise | 2 | baseline_raw | 0.0004 | 0.0030 | 1688.8596 | 18 |
| noise | 2 | pipeline_basic | 0.0040 | 0.0312 | 1404.5753 | 18 |
| noise | 2 | pipeline_clahe | 0.0068 | 0.0805 | 3797.9174 | 18 |
| noise | 2 | pipeline_full | 0.0083 | 0.0994 | 3517.3934 | 18 |
| noise | 2 | pipeline_full_otsu | 0.0193 | 0.1689 | 3714.2524 | 18 |
| noise | 3 | baseline_raw | 0.0351 | 0.1369 | 1684.4466 | 18 |
| noise | 3 | pipeline_basic | 0.0048 | 0.0375 | 1503.2659 | 18 |
| noise | 3 | pipeline_clahe | 0.0341 | 0.1618 | 3651.2737 | 18 |
| noise | 3 | pipeline_full | 0.0074 | 0.0789 | 4186.1690 | 18 |
| noise | 3 | pipeline_full_otsu | 0.0203 | 0.1711 | 3709.4482 | 18 |
| none | 0 | baseline_raw | 0.0003 | 0.0025 | 1438.5064 | 18 |
| none | 0 | pipeline_basic | 0.0020 | 0.0161 | 1418.3244 | 18 |
| none | 0 | pipeline_clahe | 0.0077 | 0.0910 | 3647.2077 | 18 |
| none | 0 | pipeline_full | 0.0084 | 0.0988 | 3813.2619 | 18 |
| none | 0 | pipeline_full_otsu | 0.0232 | 0.2139 | 3961.7236 | 18 |
| skew | 1 | baseline_raw | 0.0010 | 0.0073 | 1372.1397 | 18 |
| skew | 1 | pipeline_basic | 0.0067 | 0.0472 | 1326.0321 | 18 |
| skew | 1 | pipeline_clahe | 0.0032 | 0.0345 | 4239.1564 | 18 |
| skew | 1 | pipeline_full | 0.0061 | 0.0632 | 3165.3345 | 18 |
| skew | 1 | pipeline_full_otsu | 0.0178 | 0.1439 | 3058.6722 | 18 |
| skew | 2 | baseline_raw | 0.0103 | 0.0597 | 1356.4543 | 18 |
| skew | 2 | pipeline_basic | 0.0193 | 0.1163 | 1311.4617 | 18 |
| skew | 2 | pipeline_clahe | 0.0035 | 0.0287 | 3821.6438 | 18 |
| skew | 2 | pipeline_full | 0.0048 | 0.0390 | 4179.0370 | 18 |
| skew | 2 | pipeline_full_otsu | 0.0189 | 0.1525 | 4279.2177 | 18 |
| skew | 3 | baseline_raw | 0.0437 | 0.1787 | 1262.4096 | 18 |
| skew | 3 | pipeline_basic | 0.0652 | 0.2873 | 1296.6055 | 18 |
| skew | 3 | pipeline_clahe | 0.0051 | 0.0383 | 4006.9763 | 18 |
| skew | 3 | pipeline_full | 0.0067 | 0.0573 | 3664.0357 | 18 |
| skew | 3 | pipeline_full_otsu | 0.0204 | 0.1572 | 4492.6080 | 18 |

## Leave-one-stage-out ablation of `pipeline_full`

| ablated_pipeline | variant | removed_stage | stage_index | cer_mean | cer_median | wer_mean | delta_cer_vs_full | delta_wer_vs_full | runtime_ms_mean | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pipeline_full | full |  | -1 | 0.0089 | 0.0073 | 0.0898 | 0.0000 | 0.0000 | 4336.5637 | 288 |
| pipeline_full | minus_00_grayscale | grayscale | 0 | 0.0089 | 0.0073 | 0.0898 | 0.0000 | 0.0000 | 4135.8526 | 288 |
| pipeline_full | minus_01_nlmeans_denoise | nlmeans_denoise | 1 | 0.0213 | 0.0084 | 0.1320 | 0.0124 | 0.0422 | 1586.5690 | 288 |
| pipeline_full | minus_02_median_blur | median_blur | 2 | 0.0028 | 0.0012 | 0.0202 | -0.0061 | -0.0696 | 4253.8964 | 288 |
| pipeline_full | minus_03_bilateral_filter | bilateral_filter | 3 | 0.0078 | 0.0024 | 0.0511 | -0.0011 | -0.0387 | 4065.5125 | 288 |
| pipeline_full | minus_04_deskew | deskew | 4 | 0.0136 | 0.0097 | 0.1157 | 0.0047 | 0.0260 | 3869.9666 | 288 |
| pipeline_full | minus_05_adaptive_threshold | adaptive_threshold | 5 | 0.0858 | 0.0098 | 0.1758 | 0.0769 | 0.0861 | 4064.8887 | 288 |
| pipeline_full | minus_06_morphology_open | morphology_open | 6 | 0.0050 | 0.0024 | 0.0395 | -0.0039 | -0.0502 | 4175.0525 | 288 |
