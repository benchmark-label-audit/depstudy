# Paper numbers (tie-aware expected AP; item bootstrap B=2000; Holm per RQ x bench x metric)

## RQ1-gsm8k

| contrast | metric | a | b | diff [95% CI] | Holm p | equiv |
|---|---|---|---|---|---|---|
| mv: strong - full | ap | 0.280 | 0.223 | +0.057 [0.027, 0.092] | 0.017 | False |
| mv: offbench - full | ap | 0.299 | 0.223 | +0.076 [0.037, 0.117] | 0.017 | False |
| mv: offbench - strong | ap | 0.299 | 0.280 | +0.019 [-0.012, 0.052] | 1.000 | False |
| crh: strong - full | ap | 0.281 | 0.224 | +0.057 [0.027, 0.091] | 0.017 | False |
| crh: offbench - full | ap | 0.301 | 0.224 | +0.076 [0.038, 0.118] | 0.017 | False |
| crh: offbench - strong | ap | 0.301 | 0.281 | +0.019 [-0.011, 0.053] | 1.000 | False |
| ds_onecoin_nf10_lo: strong - full | ap | 0.270 | 0.238 | +0.031 [0.004, 0.054] | 0.255 | False |
| ds_onecoin_nf10_lo: offbench - full | ap | 0.300 | 0.238 | +0.062 [0.018, 0.103] | 0.068 | False |
| ds_onecoin_nf10_lo: offbench - strong | ap | 0.300 | 0.270 | +0.030 [-0.009, 0.073] | 1.000 | False |
| glad_lo: strong - full | ap | 0.268 | 0.244 | +0.024 [0.000, 0.048] | 0.700 | False |
| glad_lo: offbench - full | ap | 0.300 | 0.244 | +0.056 [0.016, 0.098] | 0.112 | False |
| glad_lo: offbench - strong | ap | 0.300 | 0.268 | +0.032 [-0.006, 0.075] | 1.000 | False |
| irt4pl: strong - full | ap | 0.243 | 0.261 | -0.018 [-0.085, 0.052] | 1.000 | False |
| irt4pl: offbench - full | ap | 0.210 | 0.261 | -0.051 [-0.103, 0.004] | 0.871 | False |
| irt4pl: offbench - strong | ap | 0.210 | 0.243 | -0.033 [-0.084, 0.013] | 1.000 | False |
| crh: top10 - random10 | ap | 0.280 | 0.172 | +0.108 [0.064, 0.158] | 0.017 | False |
| crh: top30 - random30 | ap | 0.307 | 0.207 | +0.101 [0.055, 0.150] | 0.017 | False |
| crh: top100 - random100 | ap | 0.301 | 0.225 | +0.075 [0.034, 0.119] | 0.017 | False |
| crh: top300 - random300 | ap | 0.284 | 0.229 | +0.055 [0.026, 0.090] | 0.017 | False |
| crh: top1000 - random1000 | ap | 0.239 | 0.224 | +0.015 [-0.001, 0.032] | 0.871 | False |
| crh: random300 - all | ap | 0.229 | 0.224 | +0.004 [-0.012, 0.025] | 1.000 | False |
| ds_onecoin_nf10_lo: top10 - random10 | ap | 0.309 | 0.194 | +0.115 [0.060, 0.172] | 0.017 | False |
| ds_onecoin_nf10_lo: top30 - random30 | ap | 0.315 | 0.234 | +0.081 [0.034, 0.127] | 0.017 | False |
| ds_onecoin_nf10_lo: top100 - random100 | ap | 0.300 | 0.236 | +0.064 [0.020, 0.104] | 0.042 | False |
| ds_onecoin_nf10_lo: top300 - random300 | ap | 0.293 | 0.241 | +0.052 [0.017, 0.090] | 0.054 | False |
| ds_onecoin_nf10_lo: top1000 - random1000 | ap | 0.249 | 0.240 | +0.009 [-0.006, 0.023] | 1.000 | False |
| ds_onecoin_nf10_lo: random300 - all | ap | 0.241 | 0.238 | +0.003 [-0.005, 0.010] | 1.000 | True |
| glad_lo: top10 - random10 | ap | 0.310 | 0.176 | +0.135 [0.069, 0.205] | 0.017 | False |
| glad_lo: top30 - random30 | ap | 0.314 | 0.209 | +0.104 [0.054, 0.162] | 0.017 | False |
| glad_lo: top100 - random100 | ap | 0.300 | 0.236 | +0.064 [0.022, 0.105] | 0.042 | False |
| glad_lo: top300 - random300 | ap | 0.298 | 0.242 | +0.056 [0.023, 0.093] | 0.042 | False |
| glad_lo: top1000 - random1000 | ap | 0.251 | 0.244 | +0.007 [-0.007, 0.022] | 1.000 | True |
| glad_lo: random300 - all | ap | 0.242 | 0.244 | -0.002 [-0.011, 0.008] | 1.000 | True |
| mv: strong - full | ap_key | 0.209 | 0.146 | +0.063 [nan, nan] | 1.000 | False |
| mv: offbench - full | ap_key | 0.327 | 0.146 | +0.181 [nan, nan] | 1.000 | False |
| mv: offbench - strong | ap_key | 0.327 | 0.209 | +0.118 [nan, nan] | 1.000 | False |
| crh: strong - full | ap_key | 0.209 | 0.148 | +0.061 [nan, nan] | 1.000 | False |
| crh: offbench - full | ap_key | 0.328 | 0.148 | +0.179 [nan, nan] | 1.000 | False |
| crh: offbench - strong | ap_key | 0.328 | 0.209 | +0.118 [nan, nan] | 1.000 | False |
| ds_onecoin_nf10_lo: strong - full | ap_key | 0.322 | 0.219 | +0.103 [nan, nan] | 1.000 | False |
| ds_onecoin_nf10_lo: offbench - full | ap_key | 0.513 | 0.219 | +0.294 [nan, nan] | 0.494 | False |
| ds_onecoin_nf10_lo: offbench - strong | ap_key | 0.513 | 0.322 | +0.191 [nan, nan] | 1.000 | False |
| glad_lo: strong - full | ap_key | 0.319 | 0.308 | +0.012 [nan, nan] | 1.000 | False |
| glad_lo: offbench - full | ap_key | 0.496 | 0.308 | +0.188 [nan, nan] | 0.713 | False |
| glad_lo: offbench - strong | ap_key | 0.496 | 0.319 | +0.176 [nan, nan] | 1.000 | False |
| irt4pl: strong - full | ap_key | 0.268 | 0.306 | -0.038 [nan, nan] | 1.000 | False |
| irt4pl: offbench - full | ap_key | 0.068 | 0.306 | -0.238 [nan, nan] | 0.713 | False |
| irt4pl: offbench - strong | ap_key | 0.068 | 0.268 | -0.200 [nan, nan] | 0.099 | False |
| crh: top10 - random10 | ap_key | 0.147 | 0.054 | +0.093 [nan, nan] | 0.186 | False |
| crh: top30 - random30 | ap_key | 0.210 | 0.069 | +0.141 [nan, nan] | 0.600 | False |
| crh: top100 - random100 | ap_key | 0.328 | 0.099 | +0.229 [nan, nan] | 0.480 | False |
| crh: top300 - random300 | ap_key | 0.214 | 0.109 | +0.105 [nan, nan] | 0.480 | False |
| crh: top1000 - random1000 | ap_key | 0.175 | 0.128 | +0.047 [nan, nan] | 1.000 | False |
| crh: random300 - all | ap_key | 0.109 | 0.148 | -0.040 [nan, nan] | 1.000 | False |
| ds_onecoin_nf10_lo: top10 - random10 | ap_key | 0.361 | 0.163 | +0.198 [nan, nan] | 1.000 | False |
| ds_onecoin_nf10_lo: top30 - random30 | ap_key | 0.492 | 0.227 | +0.265 [nan, nan] | 0.480 | False |
| ds_onecoin_nf10_lo: top100 - random100 | ap_key | 0.513 | 0.233 | +0.280 [nan, nan] | 0.480 | False |
| ds_onecoin_nf10_lo: top300 - random300 | ap_key | 0.448 | 0.239 | +0.208 [nan, nan] | 0.714 | False |
| ds_onecoin_nf10_lo: top1000 - random1000 | ap_key | 0.282 | 0.233 | +0.050 [nan, nan] | 1.000 | False |
| ds_onecoin_nf10_lo: random300 - all | ap_key | 0.239 | 0.219 | +0.021 [nan, nan] | 1.000 | False |
| glad_lo: top10 - random10 | ap_key | 0.340 | 0.142 | +0.198 [nan, nan] | 1.000 | False |
| glad_lo: top30 - random30 | ap_key | 0.453 | 0.176 | +0.277 [nan, nan] | 0.128 | False |
| glad_lo: top100 - random100 | ap_key | 0.496 | 0.245 | +0.251 [nan, nan] | 0.500 | False |
| glad_lo: top300 - random300 | ap_key | 0.481 | 0.267 | +0.214 [nan, nan] | 0.720 | False |
| glad_lo: top1000 - random1000 | ap_key | 0.316 | 0.304 | +0.012 [nan, nan] | 1.000 | False |
| glad_lo: random300 - all | ap_key | 0.267 | 0.308 | -0.040 [nan, nan] | 1.000 | False |

## RQ1-hellaswag

| contrast | metric | a | b | diff [95% CI] | Holm p | equiv |
|---|---|---|---|---|---|---|
| mv: strong - full | ap | 0.101 | 0.089 | +0.013 [0.004, 0.020] | 0.093 | True |
| mv: offbench - full | ap | 0.089 | 0.089 | +0.001 [-0.007, 0.006] | 1.000 | True |
| mv: offbench - strong | ap | 0.089 | 0.101 | -0.012 [-0.020, -0.005] | 0.064 | True |
| crh: strong - full | ap | 0.101 | 0.090 | +0.011 [0.003, 0.019] | 0.261 | True |
| crh: offbench - full | ap | 0.089 | 0.090 | -0.000 [-0.009, 0.005] | 1.000 | True |
| crh: offbench - strong | ap | 0.089 | 0.101 | -0.012 [-0.020, -0.005] | 0.120 | True |
| ds_onecoin_lo: strong - full | ap | 0.101 | 0.092 | +0.009 [-0.002, 0.017] | 1.000 | True |
| ds_onecoin_lo: offbench - full | ap | 0.091 | 0.092 | -0.002 [-0.012, 0.005] | 1.000 | True |
| ds_onecoin_lo: offbench - strong | ap | 0.091 | 0.101 | -0.010 [-0.020, -0.001] | 0.588 | True |
| glad_lo: strong - full | ap | 0.101 | 0.093 | +0.008 [-0.001, 0.015] | 1.000 | True |
| glad_lo: offbench - full | ap | 0.091 | 0.093 | -0.002 [-0.011, 0.004] | 1.000 | True |
| glad_lo: offbench - strong | ap | 0.091 | 0.101 | -0.010 [-0.020, -0.001] | 0.625 | True |
| irt4pl: strong - full | ap | 0.078 | 0.085 | -0.007 [-0.024, 0.010] | 1.000 | False |
| irt4pl: offbench - full | ap | 0.090 | 0.085 | +0.005 [-0.011, 0.022] | 1.000 | True |
| irt4pl: offbench - strong | ap | 0.090 | 0.078 | +0.012 [-0.002, 0.027] | 1.000 | False |
| crh: top10 - random10 | ap | 0.082 | 0.087 | -0.005 [-0.010, 0.001] | 1.000 | True |
| crh: top30 - random30 | ap | 0.084 | 0.089 | -0.005 [-0.011, 0.001] | 1.000 | True |
| crh: top100 - random100 | ap | 0.089 | 0.090 | -0.000 [-0.006, 0.005] | 1.000 | True |
| crh: top300 - random300 | ap | 0.098 | 0.091 | +0.007 [0.001, 0.014] | 0.624 | True |
| crh: top1000 - random1000 | ap | 0.101 | 0.091 | +0.010 [0.005, 0.017] | 0.017 | True |
| crh: random300 - all | ap | 0.091 | 0.090 | +0.001 [-0.004, 0.003] | 1.000 | True |
| ds_onecoin_lo: top10 - random10 | ap | 0.082 | 0.088 | -0.005 [-0.011, 0.000] | 1.000 | True |
| ds_onecoin_lo: top30 - random30 | ap | 0.085 | 0.089 | -0.004 [-0.010, 0.002] | 1.000 | True |
| ds_onecoin_lo: top100 - random100 | ap | 0.091 | 0.091 | -0.000 [-0.007, 0.006] | 1.000 | True |
| ds_onecoin_lo: top300 - random300 | ap | 0.093 | 0.092 | +0.001 [-0.006, 0.007] | 1.000 | True |
| ds_onecoin_lo: top1000 - random1000 | ap | 0.098 | 0.092 | +0.006 [0.000, 0.011] | 0.966 | True |
| ds_onecoin_lo: random300 - all | ap | 0.092 | 0.092 | -0.000 [-0.005, 0.004] | 1.000 | True |
| glad_lo: top10 - random10 | ap | 0.082 | 0.089 | -0.007 [-0.014, -0.001] | 0.588 | True |
| glad_lo: top30 - random30 | ap | 0.085 | 0.090 | -0.006 [-0.013, 0.000] | 1.000 | True |
| glad_lo: top100 - random100 | ap | 0.091 | 0.093 | -0.002 [-0.010, 0.004] | 1.000 | True |
| glad_lo: top300 - random300 | ap | 0.097 | 0.092 | +0.005 [-0.002, 0.014] | 1.000 | True |
| glad_lo: top1000 - random1000 | ap | 0.100 | 0.094 | +0.006 [0.001, 0.012] | 0.792 | True |
| glad_lo: random300 - all | ap | 0.092 | 0.093 | -0.001 [-0.005, 0.003] | 1.000 | True |
| mv: strong - full | ap_key | 0.147 | 0.121 | +0.025 [0.013, 0.038] | 0.017 | False |
| mv: offbench - full | ap_key | 0.126 | 0.121 | +0.004 [-0.005, 0.012] | 1.000 | True |
| mv: offbench - strong | ap_key | 0.126 | 0.147 | -0.021 [-0.034, -0.010] | 0.017 | False |
| crh: strong - full | ap_key | 0.146 | 0.123 | +0.023 [0.011, 0.035] | 0.017 | False |
| crh: offbench - full | ap_key | 0.126 | 0.123 | +0.003 [-0.007, 0.011] | 1.000 | True |
| crh: offbench - strong | ap_key | 0.126 | 0.146 | -0.020 [-0.033, -0.009] | 0.017 | False |
| ds_onecoin_lo: strong - full | ap_key | 0.147 | 0.129 | +0.018 [0.004, 0.033] | 0.308 | False |
| ds_onecoin_lo: offbench - full | ap_key | 0.127 | 0.129 | -0.002 [-0.014, 0.007] | 1.000 | True |
| ds_onecoin_lo: offbench - strong | ap_key | 0.127 | 0.147 | -0.020 [-0.035, -0.007] | 0.027 | False |
| glad_lo: strong - full | ap_key | 0.146 | 0.129 | +0.017 [0.006, 0.029] | 0.052 | False |
| glad_lo: offbench - full | ap_key | 0.126 | 0.129 | -0.002 [-0.013, 0.006] | 1.000 | True |
| glad_lo: offbench - strong | ap_key | 0.126 | 0.146 | -0.020 [-0.035, -0.006] | 0.075 | False |
| irt4pl: strong - full | ap_key | 0.109 | 0.129 | -0.020 [-0.048, 0.007] | 1.000 | False |
| irt4pl: offbench - full | ap_key | 0.127 | 0.129 | -0.001 [-0.027, 0.023] | 1.000 | False |
| irt4pl: offbench - strong | ap_key | 0.127 | 0.109 | +0.019 [-0.000, 0.039] | 1.000 | False |
| crh: top10 - random10 | ap_key | 0.113 | 0.120 | -0.007 [-0.014, 0.000] | 1.000 | True |
| crh: top30 - random30 | ap_key | 0.117 | 0.123 | -0.006 [-0.014, 0.002] | 1.000 | True |
| crh: top100 - random100 | ap_key | 0.126 | 0.124 | +0.002 [-0.006, 0.009] | 1.000 | True |
| crh: top300 - random300 | ap_key | 0.137 | 0.125 | +0.012 [0.004, 0.021] | 0.096 | True |
| crh: top1000 - random1000 | ap_key | 0.142 | 0.124 | +0.018 [0.011, 0.027] | 0.017 | False |
| crh: random300 - all | ap_key | 0.125 | 0.123 | +0.002 [-0.003, 0.004] | 1.000 | True |
| ds_onecoin_lo: top10 - random10 | ap_key | 0.112 | 0.120 | -0.008 [-0.017, -0.000] | 0.960 | True |
| ds_onecoin_lo: top30 - random30 | ap_key | 0.117 | 0.122 | -0.005 [-0.014, 0.003] | 1.000 | True |
| ds_onecoin_lo: top100 - random100 | ap_key | 0.127 | 0.126 | +0.001 [-0.009, 0.009] | 1.000 | True |
| ds_onecoin_lo: top300 - random300 | ap_key | 0.130 | 0.128 | +0.002 [-0.007, 0.010] | 1.000 | True |
| ds_onecoin_lo: top1000 - random1000 | ap_key | 0.138 | 0.127 | +0.011 [0.004, 0.017] | 0.017 | True |
| ds_onecoin_lo: random300 - all | ap_key | 0.128 | 0.129 | -0.001 [-0.006, 0.004] | 1.000 | True |
| glad_lo: top10 - random10 | ap_key | 0.112 | 0.123 | -0.011 [-0.020, -0.002] | 0.308 | True |
| glad_lo: top30 - random30 | ap_key | 0.117 | 0.125 | -0.008 [-0.018, 0.000] | 1.000 | True |
| glad_lo: top100 - random100 | ap_key | 0.126 | 0.129 | -0.002 [-0.013, 0.006] | 1.000 | True |
| glad_lo: top300 - random300 | ap_key | 0.134 | 0.128 | +0.006 [-0.003, 0.016] | 1.000 | True |
| glad_lo: top1000 - random1000 | ap_key | 0.140 | 0.130 | +0.010 [0.003, 0.017] | 0.115 | True |
| glad_lo: random300 - all | ap_key | 0.128 | 0.129 | -0.001 [-0.006, 0.003] | 1.000 | True |

## RQ1-mmlu

| contrast | metric | a | b | diff [95% CI] | Holm p | equiv |
|---|---|---|---|---|---|---|
| mv: strong - full | ap | 0.221 | 0.141 | +0.080 [0.049, 0.109] | 0.017 | False |
| mv: offbench - full | ap | 0.178 | 0.141 | +0.037 [0.015, 0.057] | 0.038 | False |
| mv: offbench - strong | ap | 0.178 | 0.221 | -0.043 [-0.067, -0.020] | 0.017 | False |
| crh: strong - full | ap | 0.225 | 0.145 | +0.080 [0.048, 0.110] | 0.017 | False |
| crh: offbench - full | ap | 0.177 | 0.145 | +0.032 [0.010, 0.053] | 0.054 | False |
| crh: offbench - strong | ap | 0.177 | 0.225 | -0.049 [-0.073, -0.025] | 0.017 | False |
| ds_onecoin_lo: strong - full | ap | 0.223 | 0.190 | +0.032 [0.005, 0.059] | 0.448 | False |
| ds_onecoin_lo: offbench - full | ap | 0.184 | 0.190 | -0.006 [-0.030, 0.016] | 1.000 | False |
| ds_onecoin_lo: offbench - strong | ap | 0.184 | 0.223 | -0.039 [-0.064, -0.014] | 0.022 | False |
| glad_lo: strong - full | ap | 0.219 | 0.191 | +0.027 [-0.001, 0.055] | 0.728 | False |
| glad_lo: offbench - full | ap | 0.179 | 0.191 | -0.012 [-0.034, 0.009] | 1.000 | False |
| glad_lo: offbench - strong | ap | 0.179 | 0.219 | -0.040 [-0.067, -0.015] | 0.022 | False |
| irt4pl: strong - full | ap | 0.191 | 0.119 | +0.073 [0.043, 0.105] | 0.017 | False |
| irt4pl: offbench - full | ap | 0.142 | 0.119 | +0.023 [-0.000, 0.049] | 0.728 | False |
| irt4pl: offbench - strong | ap | 0.142 | 0.191 | -0.049 [-0.083, -0.019] | 0.054 | False |
| crh: top10 - random10 | ap | 0.181 | 0.113 | +0.068 [0.047, 0.090] | 0.017 | False |
| crh: top30 - random30 | ap | 0.179 | 0.134 | +0.045 [0.025, 0.067] | 0.017 | False |
| crh: top100 - random100 | ap | 0.177 | 0.138 | +0.039 [0.018, 0.059] | 0.017 | False |
| crh: top300 - random300 | ap | 0.185 | 0.144 | +0.040 [0.018, 0.062] | 0.022 | False |
| crh: top1000 - random1000 | ap | 0.204 | 0.144 | +0.061 [0.035, 0.088] | 0.017 | False |
| crh: random300 - all | ap | 0.144 | 0.145 | -0.001 [-0.007, 0.004] | 1.000 | True |
| ds_onecoin_lo: top10 - random10 | ap | 0.195 | 0.129 | +0.066 [0.041, 0.093] | 0.017 | False |
| ds_onecoin_lo: top30 - random30 | ap | 0.185 | 0.163 | +0.022 [-0.000, 0.044] | 0.728 | False |
| ds_onecoin_lo: top100 - random100 | ap | 0.184 | 0.182 | +0.002 [-0.021, 0.024] | 1.000 | False |
| ds_onecoin_lo: top300 - random300 | ap | 0.191 | 0.189 | +0.002 [-0.020, 0.023] | 1.000 | False |
| ds_onecoin_lo: top1000 - random1000 | ap | 0.207 | 0.189 | +0.018 [0.001, 0.035] | 0.540 | False |
| ds_onecoin_lo: random300 - all | ap | 0.189 | 0.190 | -0.002 [-0.005, 0.002] | 1.000 | True |
| glad_lo: top10 - random10 | ap | 0.192 | 0.129 | +0.063 [0.037, 0.090] | 0.017 | False |
| glad_lo: top30 - random30 | ap | 0.182 | 0.165 | +0.016 [-0.007, 0.038] | 1.000 | False |
| glad_lo: top100 - random100 | ap | 0.179 | 0.180 | -0.002 [-0.024, 0.020] | 1.000 | False |
| glad_lo: top300 - random300 | ap | 0.184 | 0.189 | -0.005 [-0.024, 0.015] | 1.000 | False |
| glad_lo: top1000 - random1000 | ap | 0.196 | 0.189 | +0.007 [-0.006, 0.022] | 1.000 | True |
| glad_lo: random300 - all | ap | 0.189 | 0.191 | -0.002 [-0.007, 0.003] | 1.000 | True |
| mv: strong - full | ap_key | 0.302 | 0.134 | +0.169 [0.100, 0.240] | 0.017 | False |
| mv: offbench - full | ap_key | 0.206 | 0.134 | +0.072 [0.020, 0.115] | 0.100 | False |
| mv: offbench - strong | ap_key | 0.206 | 0.302 | -0.097 [-0.161, -0.042] | 0.017 | False |
| crh: strong - full | ap_key | 0.315 | 0.142 | +0.173 [0.103, 0.245] | 0.017 | False |
| crh: offbench - full | ap_key | 0.205 | 0.142 | +0.063 [0.008, 0.107] | 0.391 | False |
| crh: offbench - strong | ap_key | 0.205 | 0.315 | -0.110 [-0.175, -0.055] | 0.017 | False |
| ds_onecoin_lo: strong - full | ap_key | 0.313 | 0.246 | +0.067 [-0.006, 0.130] | 1.000 | False |
| ds_onecoin_lo: offbench - full | ap_key | 0.230 | 0.246 | -0.016 [-0.085, 0.040] | 1.000 | False |
| ds_onecoin_lo: offbench - strong | ap_key | 0.230 | 0.313 | -0.083 [-0.151, -0.020] | 0.144 | False |
| glad_lo: strong - full | ap_key | 0.303 | 0.256 | +0.047 [-0.018, 0.112] | 1.000 | False |
| glad_lo: offbench - full | ap_key | 0.217 | 0.256 | -0.039 [-0.095, 0.015] | 1.000 | False |
| glad_lo: offbench - strong | ap_key | 0.217 | 0.303 | -0.086 [-0.154, -0.022] | 0.114 | False |
| irt4pl: strong - full | ap_key | 0.214 | 0.086 | +0.128 [0.071, 0.198] | 0.017 | False |
| irt4pl: offbench - full | ap_key | 0.109 | 0.086 | +0.024 [-0.011, 0.052] | 1.000 | False |
| irt4pl: offbench - strong | ap_key | 0.109 | 0.214 | -0.104 [-0.174, -0.049] | 0.017 | False |
| crh: top10 - random10 | ap_key | 0.176 | 0.077 | +0.100 [0.070, 0.133] | 0.017 | False |
| crh: top30 - random30 | ap_key | 0.196 | 0.115 | +0.080 [0.041, 0.119] | 0.017 | False |
| crh: top100 - random100 | ap_key | 0.205 | 0.124 | +0.080 [0.039, 0.121] | 0.017 | False |
| crh: top300 - random300 | ap_key | 0.226 | 0.139 | +0.088 [0.039, 0.136] | 0.021 | False |
| crh: top1000 - random1000 | ap_key | 0.297 | 0.138 | +0.159 [0.089, 0.227] | 0.017 | False |
| crh: random300 - all | ap_key | 0.139 | 0.142 | -0.003 [-0.021, 0.009] | 1.000 | True |
| ds_onecoin_lo: top10 - random10 | ap_key | 0.226 | 0.109 | +0.118 [0.069, 0.167] | 0.017 | False |
| ds_onecoin_lo: top30 - random30 | ap_key | 0.227 | 0.174 | +0.053 [-0.002, 0.104] | 1.000 | False |
| ds_onecoin_lo: top100 - random100 | ap_key | 0.230 | 0.227 | +0.003 [-0.060, 0.057] | 1.000 | False |
| ds_onecoin_lo: top300 - random300 | ap_key | 0.241 | 0.243 | -0.002 [-0.061, 0.052] | 1.000 | False |
| ds_onecoin_lo: top1000 - random1000 | ap_key | 0.280 | 0.243 | +0.037 [-0.012, 0.080] | 1.000 | False |
| ds_onecoin_lo: random300 - all | ap_key | 0.243 | 0.246 | -0.003 [-0.013, 0.006] | 1.000 | True |
| glad_lo: top10 - random10 | ap_key | 0.222 | 0.108 | +0.113 [0.063, 0.166] | 0.017 | False |
| glad_lo: top30 - random30 | ap_key | 0.219 | 0.178 | +0.041 [-0.017, 0.096] | 1.000 | False |
| glad_lo: top100 - random100 | ap_key | 0.217 | 0.223 | -0.006 [-0.062, 0.048] | 1.000 | False |
| glad_lo: top300 - random300 | ap_key | 0.225 | 0.248 | -0.023 [-0.071, 0.024] | 1.000 | False |
| glad_lo: top1000 - random1000 | ap_key | 0.260 | 0.251 | +0.009 [-0.028, 0.049] | 1.000 | False |
| glad_lo: random300 - all | ap_key | 0.248 | 0.256 | -0.008 [-0.023, 0.006] | 1.000 | False |

## RQ1-sources-gsm8k

| contrast | metric | a | b | diff [95% CI] | Holm p | equiv |
|---|---|---|---|---|---|---|
| mv: helm_all - v1_strong_k | ap | 0.392 | 0.259 | +0.133 [0.073, 0.199] | 0.013 | False |
| mv: helm_all - v1_random_k | ap | 0.392 | 0.193 | +0.198 [0.131, 0.270] | 0.013 | False |
| mv: helm_plus_v1_all - helm_all | ap | 0.234 | 0.392 | -0.158 [-0.222, -0.100] | 0.013 | False |
| mv: helm_plus_v1_strong100 - helm_all | ap | 0.338 | 0.392 | -0.054 [-0.090, -0.022] | 0.013 | False |
| mv: vendor_concentrated - vendor_diverse | ap | 0.407 | 0.360 | +0.048 [0.014, 0.087] | 0.021 | False |
| crh: helm_all - v1_strong_k | ap | 0.403 | 0.259 | +0.144 [0.081, 0.213] | 0.013 | False |
| crh: helm_all - v1_random_k | ap | 0.403 | 0.194 | +0.208 [0.140, 0.280] | 0.013 | False |
| crh: helm_plus_v1_all - helm_all | ap | 0.235 | 0.403 | -0.167 [-0.234, -0.108] | 0.013 | False |
| crh: helm_plus_v1_strong100 - helm_all | ap | 0.345 | 0.403 | -0.058 [-0.095, -0.025] | 0.013 | False |
| crh: vendor_concentrated - vendor_diverse | ap | 0.420 | 0.375 | +0.046 [0.011, 0.086] | 0.045 | False |
| ds_onecoin_nf10_lo: helm_all - v1_strong_k | ap | 0.428 | 0.254 | +0.173 [0.095, 0.253] | 0.013 | False |
| ds_onecoin_nf10_lo: helm_all - v1_random_k | ap | 0.428 | 0.212 | +0.215 [0.136, 0.293] | 0.013 | False |
| ds_onecoin_nf10_lo: helm_plus_v1_all - helm_all | ap | 0.232 | 0.428 | -0.195 [-0.269, -0.122] | 0.013 | False |
| ds_onecoin_nf10_lo: helm_plus_v1_strong100 - helm_all | ap | 0.343 | 0.428 | -0.085 [-0.140, -0.035] | 0.013 | False |
| ds_onecoin_nf10_lo: vendor_concentrated - vendor_diverse | ap | 0.428 | 0.391 | +0.037 [0.003, 0.077] | 0.108 | False |
| glad_lo: helm_all - v1_strong_k | ap | 0.420 | 0.253 | +0.167 [0.090, 0.247] | 0.013 | False |
| glad_lo: helm_all - v1_random_k | ap | 0.420 | 0.209 | +0.211 [0.132, 0.287] | 0.013 | False |
| glad_lo: helm_plus_v1_all - helm_all | ap | 0.228 | 0.420 | -0.192 [-0.266, -0.116] | 0.013 | False |
| glad_lo: helm_plus_v1_strong100 - helm_all | ap | 0.340 | 0.420 | -0.080 [-0.133, -0.030] | 0.013 | False |
| glad_lo: vendor_concentrated - vendor_diverse | ap | 0.425 | 0.385 | +0.040 [0.005, 0.080] | 0.108 | False |
| irt4pl: helm_all - v1_strong_k | ap | 0.420 | 0.223 | +0.197 [0.103, 0.285] | 0.013 | False |
| irt4pl: helm_all - v1_random_k | ap | 0.420 | 0.208 | +0.212 [0.125, 0.297] | 0.013 | False |
| irt4pl: helm_plus_v1_all - helm_all | ap | 0.327 | 0.420 | -0.093 [-0.183, -0.004] | 0.108 | False |
| irt4pl: helm_plus_v1_strong100 - helm_all | ap | 0.459 | 0.420 | +0.039 [0.013, 0.070] | 0.021 | False |
| irt4pl: vendor_concentrated - vendor_diverse | ap | 0.314 | 0.368 | -0.054 [-0.118, 0.010] | 0.108 | False |
| mv: helm_all - v1_strong_k | ap_key | 0.593 | 0.167 | +0.426 [nan, nan] | 0.160 | False |
| mv: helm_all - v1_random_k | ap_key | 0.593 | 0.078 | +0.515 [nan, nan] | 0.013 | False |
| mv: helm_plus_v1_all - helm_all | ap_key | 0.236 | 0.593 | -0.357 [nan, nan] | 0.676 | False |
| mv: helm_plus_v1_strong100 - helm_all | ap_key | 0.361 | 0.593 | -0.232 [nan, nan] | 0.828 | False |
| mv: vendor_concentrated - vendor_diverse | ap_key | 0.656 | 0.532 | +0.124 [nan, nan] | 0.750 | False |
| crh: helm_all - v1_strong_k | ap_key | 0.599 | 0.167 | +0.432 [nan, nan] | 0.023 | False |
| crh: helm_all - v1_random_k | ap_key | 0.599 | 0.078 | +0.521 [nan, nan] | 0.013 | False |
| crh: helm_plus_v1_all - helm_all | ap_key | 0.236 | 0.599 | -0.363 [nan, nan] | 0.676 | False |
| crh: helm_plus_v1_strong100 - helm_all | ap_key | 0.378 | 0.599 | -0.221 [nan, nan] | 0.676 | False |
| crh: vendor_concentrated - vendor_diverse | ap_key | 0.649 | 0.550 | +0.099 [nan, nan] | 0.828 | False |
| ds_onecoin_nf10_lo: helm_all - v1_strong_k | ap_key | 0.756 | 0.241 | +0.515 [nan, nan] | 0.600 | False |
| ds_onecoin_nf10_lo: helm_all - v1_random_k | ap_key | 0.756 | 0.197 | +0.558 [nan, nan] | 0.160 | False |
| ds_onecoin_nf10_lo: helm_plus_v1_all - helm_all | ap_key | 0.271 | 0.756 | -0.484 [nan, nan] | 0.360 | False |
| ds_onecoin_nf10_lo: helm_plus_v1_strong100 - helm_all | ap_key | 0.532 | 0.756 | -0.223 [nan, nan] | 0.828 | False |
| ds_onecoin_nf10_lo: vendor_concentrated - vendor_diverse | ap_key | 0.747 | 0.694 | +0.053 [nan, nan] | 1.000 | False |
| glad_lo: helm_all - v1_strong_k | ap_key | 0.756 | 0.226 | +0.530 [nan, nan] | 0.360 | False |
| glad_lo: helm_all - v1_random_k | ap_key | 0.756 | 0.190 | +0.566 [nan, nan] | 0.084 | False |
| glad_lo: helm_plus_v1_all - helm_all | ap_key | 0.263 | 0.756 | -0.493 [nan, nan] | 0.360 | False |
| glad_lo: helm_plus_v1_strong100 - helm_all | ap_key | 0.512 | 0.756 | -0.244 [nan, nan] | 0.602 | False |
| glad_lo: vendor_concentrated - vendor_diverse | ap_key | 0.731 | 0.699 | +0.032 [nan, nan] | 1.000 | False |
| irt4pl: helm_all - v1_strong_k | ap_key | 0.718 | 0.290 | +0.427 [nan, nan] | 0.828 | False |
| irt4pl: helm_all - v1_random_k | ap_key | 0.718 | 0.143 | +0.575 [nan, nan] | 0.066 | False |
| irt4pl: helm_plus_v1_all - helm_all | ap_key | 0.449 | 0.718 | -0.269 [nan, nan] | 0.828 | False |
| irt4pl: helm_plus_v1_strong100 - helm_all | ap_key | 0.786 | 0.718 | +0.069 [nan, nan] | 1.000 | False |
| irt4pl: vendor_concentrated - vendor_diverse | ap_key | 0.374 | 0.497 | -0.123 [nan, nan] | 0.896 | False |

## RQ1-sources-mmlu

| contrast | metric | a | b | diff [95% CI] | Holm p | equiv |
|---|---|---|---|---|---|---|
| mv: helm_all - v1_strong_k | ap | 0.270 | 0.224 | +0.046 [0.016, 0.078] | 0.036 | False |
| mv: helm_all - v1_random_k | ap | 0.270 | 0.138 | +0.132 [0.098, 0.167] | 0.010 | False |
| mv: helm_plus_v1_all - helm_all | ap | 0.152 | 0.270 | -0.118 [-0.150, -0.086] | 0.010 | False |
| mv: helm_plus_v1_strong100 - helm_all | ap | 0.266 | 0.270 | -0.004 [-0.021, 0.012] | 0.948 | True |
| mv: vendor_concentrated - vendor_diverse | ap | 0.239 | 0.220 | +0.019 [0.003, 0.036] | 0.154 | False |
| crh: helm_all - v1_strong_k | ap | 0.274 | 0.229 | +0.045 [0.015, 0.076] | 0.030 | False |
| crh: helm_all - v1_random_k | ap | 0.274 | 0.142 | +0.132 [0.098, 0.167] | 0.010 | False |
| crh: helm_plus_v1_all - helm_all | ap | 0.155 | 0.274 | -0.118 [-0.151, -0.086] | 0.010 | False |
| crh: helm_plus_v1_strong100 - helm_all | ap | 0.267 | 0.274 | -0.007 [-0.024, 0.011] | 0.948 | False |
| crh: vendor_concentrated - vendor_diverse | ap | 0.242 | 0.224 | +0.017 [0.002, 0.034] | 0.162 | False |
| ds_onecoin_lo: helm_all - v1_strong_k | ap | 0.269 | 0.224 | +0.045 [0.020, 0.070] | 0.010 | False |
| ds_onecoin_lo: helm_all - v1_random_k | ap | 0.269 | 0.177 | +0.092 [0.064, 0.118] | 0.010 | False |
| ds_onecoin_lo: helm_plus_v1_all - helm_all | ap | 0.196 | 0.269 | -0.073 [-0.097, -0.046] | 0.010 | False |
| ds_onecoin_lo: helm_plus_v1_strong100 - helm_all | ap | 0.257 | 0.269 | -0.011 [-0.023, 0.001] | 0.240 | False |
| ds_onecoin_lo: vendor_concentrated - vendor_diverse | ap | 0.262 | 0.244 | +0.018 [0.001, 0.036] | 0.195 | False |
| glad_lo: helm_all - v1_strong_k | ap | 0.260 | 0.218 | +0.041 [0.017, 0.069] | 0.010 | False |
| glad_lo: helm_all - v1_random_k | ap | 0.260 | 0.176 | +0.084 [0.056, 0.115] | 0.010 | False |
| glad_lo: helm_plus_v1_all - helm_all | ap | 0.196 | 0.260 | -0.063 [-0.094, -0.037] | 0.010 | False |
| glad_lo: helm_plus_v1_strong100 - helm_all | ap | 0.246 | 0.260 | -0.014 [-0.028, 0.000] | 0.216 | False |
| glad_lo: vendor_concentrated - vendor_diverse | ap | 0.260 | 0.237 | +0.023 [0.006, 0.039] | 0.072 | False |
| mv: helm_all - v1_strong_k | ap_key | 0.402 | 0.302 | +0.100 [0.027, 0.175] | 0.098 | False |
| mv: helm_all - v1_random_k | ap_key | 0.402 | 0.125 | +0.277 [0.199, 0.356] | 0.013 | False |
| mv: helm_plus_v1_all - helm_all | ap_key | 0.156 | 0.402 | -0.246 [-0.322, -0.170] | 0.013 | False |
| mv: helm_plus_v1_strong100 - helm_all | ap_key | 0.419 | 0.402 | +0.017 [-0.024, 0.056] | 1.000 | False |
| mv: vendor_concentrated - vendor_diverse | ap_key | 0.311 | 0.270 | +0.041 [0.005, 0.079] | 0.260 | False |
| crh: helm_all - v1_strong_k | ap_key | 0.412 | 0.315 | +0.097 [0.025, 0.170] | 0.132 | False |
| crh: helm_all - v1_random_k | ap_key | 0.412 | 0.133 | +0.279 [0.203, 0.358] | 0.013 | False |
| crh: helm_plus_v1_all - helm_all | ap_key | 0.163 | 0.412 | -0.249 [-0.325, -0.171] | 0.013 | False |
| crh: helm_plus_v1_strong100 - helm_all | ap_key | 0.422 | 0.412 | +0.010 [-0.034, 0.053] | 1.000 | False |
| crh: vendor_concentrated - vendor_diverse | ap_key | 0.315 | 0.280 | +0.036 [0.001, 0.072] | 0.333 | False |
| ds_onecoin_lo: helm_all - v1_strong_k | ap_key | 0.398 | 0.309 | +0.088 [0.031, 0.147] | 0.064 | False |
| ds_onecoin_lo: helm_all - v1_random_k | ap_key | 0.398 | 0.214 | +0.184 [0.123, 0.242] | 0.013 | False |
| ds_onecoin_lo: helm_plus_v1_all - helm_all | ap_key | 0.258 | 0.398 | -0.140 [-0.195, -0.078] | 0.013 | False |
| ds_onecoin_lo: helm_plus_v1_strong100 - helm_all | ap_key | 0.391 | 0.398 | -0.006 [-0.035, 0.023] | 1.000 | False |
| ds_onecoin_lo: vendor_concentrated - vendor_diverse | ap_key | 0.369 | 0.328 | +0.041 [0.003, 0.082] | 0.333 | False |
| glad_lo: helm_all - v1_strong_k | ap_key | 0.370 | 0.297 | +0.073 [0.017, 0.138] | 0.132 | False |
| glad_lo: helm_all - v1_random_k | ap_key | 0.370 | 0.213 | +0.156 [0.091, 0.226] | 0.013 | False |
| glad_lo: helm_plus_v1_all - helm_all | ap_key | 0.269 | 0.370 | -0.101 [-0.172, -0.041] | 0.017 | False |
| glad_lo: helm_plus_v1_strong100 - helm_all | ap_key | 0.359 | 0.370 | -0.010 [-0.043, 0.022] | 1.000 | False |
| glad_lo: vendor_concentrated - vendor_diverse | ap_key | 0.363 | 0.315 | +0.048 [0.014, 0.085] | 0.075 | False |
| irt4pl: helm_all - v1_strong_k | ap_key | 0.181 | 0.171 | +0.010 [-0.052, 0.072] | 1.000 | False |
| irt4pl: helm_all - v1_random_k | ap_key | 0.181 | 0.072 | +0.109 [0.068, 0.159] | 0.013 | False |
| irt4pl: helm_plus_v1_all - helm_all | ap_key | 0.111 | 0.181 | -0.070 [-0.121, -0.019] | 0.104 | False |
| irt4pl: helm_plus_v1_strong100 - helm_all | ap_key | 0.189 | 0.181 | +0.007 [-0.029, 0.045] | 1.000 | False |
| irt4pl: vendor_concentrated - vendor_diverse | ap_key | 0.103 | 0.123 | -0.020 [-0.049, 0.004] | 0.616 | False |

## RQ3-gsm8k

| contrast | metric | a | b | diff [95% CI] | Holm p | equiv |
|---|---|---|---|---|---|---|
| full: ds_onecoin_nf10_lo - crh | ap | 0.238 | 0.224 | +0.014 [-0.029, 0.062] | 1.000 | False |
| full: ds_onecoin_nf10_lo - irt4pl | ap | 0.238 | 0.261 | -0.023 [-0.076, 0.032] | 1.000 | False |
| full: glad_lo - ds_onecoin_nf10_lo | ap | 0.244 | 0.238 | +0.006 [-0.006, 0.017] | 1.000 | True |
| full: catd - crh | ap | 0.236 | 0.224 | +0.011 [0.003, 0.022] | 0.224 | True |
| full: crh - mv | ap | 0.224 | 0.223 | +0.001 [-0.001, 0.002] | 1.000 | True |
| full: entropy - mv | ap | 0.158 | 0.223 | -0.065 [-0.125, -0.008] | 0.496 | False |
| full: crowdfm - ds_onecoin_nf10_lo | ap | 0.121 | 0.238 | -0.117 [-0.191, -0.054] | 0.019 | False |
| full: ensemble - ds_onecoin_nf10_lo | ap | 0.254 | 0.238 | +0.016 [-0.022, 0.051] | 1.000 | False |
| full: ensemble - crh | ap | 0.254 | 0.224 | +0.030 [0.007, 0.058] | 0.496 | False |
| full: ensemble - irt4pl | ap | 0.254 | 0.261 | -0.007 [-0.038, 0.025] | 1.000 | False |
| full: ds_onecoin_nf10_lo - ds_onecoin_lo | ap | 0.238 | 0.232 | +0.006 [-0.003, 0.019] | 1.000 | True |
| full: irt4pl_miss - irt4pl | ap | 0.236 | 0.261 | -0.025 [-0.070, 0.020] | 1.000 | False |
| small_strong: ds_onecoin_nf10_lo - crh | ap | 0.270 | 0.281 | -0.012 [-0.064, 0.036] | 1.000 | False |
| small_strong: ds_onecoin_nf10_lo - irt4pl | ap | 0.270 | 0.243 | +0.027 [-0.049, 0.099] | 1.000 | False |
| small_strong: glad_lo - ds_onecoin_nf10_lo | ap | 0.268 | 0.270 | -0.002 [-0.007, 0.002] | 1.000 | True |
| small_strong: catd - crh | ap | 0.284 | 0.281 | +0.003 [-0.000, 0.006] | 1.000 | True |
| small_strong: crh - mv | ap | 0.281 | 0.280 | +0.001 [0.000, 0.003] | 0.962 | True |
| small_strong: entropy - mv | ap | 0.177 | 0.280 | -0.103 [-0.176, -0.039] | 0.019 | False |
| small_strong: crowdfm - ds_onecoin_nf10_lo | ap | 0.204 | 0.270 | -0.065 [-0.123, 0.001] | 1.000 | False |
| small_strong: ensemble - ds_onecoin_nf10_lo | ap | 0.313 | 0.270 | +0.043 [-0.003, 0.092] | 1.000 | False |
| small_strong: ensemble - crh | ap | 0.313 | 0.281 | +0.032 [-0.002, 0.062] | 1.000 | False |
| small_strong: ensemble - irt4pl | ap | 0.313 | 0.243 | +0.070 [0.010, 0.127] | 0.672 | False |
| small_strong: accu_depen_lo - accu_nodep_lo | ap | 0.250 | 0.270 | -0.020 [-0.036, -0.002] | 0.783 | False |
| small_strong: ds_onecoin_nf10_lo - ds_onecoin_lo | ap | 0.270 | 0.231 | +0.038 [0.006, 0.067] | 0.496 | False |
| small_strong: irt4pl_miss - irt4pl | ap | 0.305 | 0.243 | +0.063 [0.018, 0.102] | 0.136 | False |
| offbench_strong: ds_onecoin_nf10_lo - crh | ap | 0.300 | 0.301 | -0.001 [-0.055, 0.054] | 1.000 | False |
| offbench_strong: ds_onecoin_nf10_lo - irt4pl | ap | 0.300 | 0.210 | +0.090 [0.027, 0.150] | 0.165 | False |
| offbench_strong: glad_lo - ds_onecoin_nf10_lo | ap | 0.300 | 0.300 | -0.000 [-0.009, 0.009] | 1.000 | True |
| offbench_strong: catd - crh | ap | 0.306 | 0.301 | +0.005 [-0.001, 0.011] | 1.000 | True |
| offbench_strong: crh - mv | ap | 0.301 | 0.299 | +0.001 [-0.001, 0.003] | 1.000 | True |
| offbench_strong: entropy - mv | ap | 0.174 | 0.299 | -0.125 [-0.200, -0.053] | 0.036 | False |
| offbench_strong: crowdfm - ds_onecoin_nf10_lo | ap | 0.232 | 0.300 | -0.068 [-0.136, -0.001] | 1.000 | False |
| offbench_strong: ensemble - ds_onecoin_nf10_lo | ap | 0.314 | 0.300 | +0.014 [-0.029, 0.055] | 1.000 | False |
| offbench_strong: ensemble - crh | ap | 0.314 | 0.301 | +0.014 [-0.026, 0.048] | 1.000 | False |
| offbench_strong: ensemble - irt4pl | ap | 0.314 | 0.210 | +0.104 [0.053, 0.154] | 0.036 | False |
| offbench_strong: accu_depen_lo - accu_nodep_lo | ap | 0.280 | 0.300 | -0.020 [-0.043, 0.003] | 1.000 | False |
| offbench_strong: ds_onecoin_nf10_lo - ds_onecoin_lo | ap | 0.300 | 0.278 | +0.022 [-0.015, 0.054] | 1.000 | False |
| offbench_strong: irt4pl_miss - irt4pl | ap | 0.181 | 0.210 | -0.029 [-0.078, 0.020] | 1.000 | False |
| full: ds_onecoin_nf10_lo - crh | ap_key | 0.219 | 0.148 | +0.071 [nan, nan] | 1.000 | False |
| full: ds_onecoin_nf10_lo - irt4pl | ap_key | 0.219 | 0.306 | -0.087 [nan, nan] | 1.000 | False |
| full: glad_lo - ds_onecoin_nf10_lo | ap_key | 0.308 | 0.219 | +0.089 [nan, nan] | 1.000 | False |
| full: catd - crh | ap_key | 0.170 | 0.148 | +0.021 [nan, nan] | 1.000 | False |
| full: crh - mv | ap_key | 0.148 | 0.146 | +0.002 [nan, nan] | 1.000 | False |
| full: entropy - mv | ap_key | 0.029 | 0.146 | -0.117 [nan, nan] | 1.000 | False |
| full: crowdfm - ds_onecoin_nf10_lo | ap_key | 0.061 | 0.219 | -0.158 [nan, nan] | 1.000 | False |
| full: ensemble - ds_onecoin_nf10_lo | ap_key | 0.269 | 0.219 | +0.050 [nan, nan] | 1.000 | False |
| full: ensemble - crh | ap_key | 0.269 | 0.148 | +0.121 [nan, nan] | 0.589 | False |
| full: ensemble - irt4pl | ap_key | 0.269 | 0.306 | -0.037 [nan, nan] | 1.000 | False |
| full: ds_onecoin_nf10_lo - ds_onecoin_lo | ap_key | 0.219 | 0.189 | +0.030 [nan, nan] | 1.000 | False |
| full: irt4pl_miss - irt4pl | ap_key | 0.136 | 0.306 | -0.170 [nan, nan] | 1.000 | False |
| small_strong: ds_onecoin_nf10_lo - crh | ap_key | 0.322 | 0.209 | +0.112 [nan, nan] | 1.000 | False |
| small_strong: ds_onecoin_nf10_lo - irt4pl | ap_key | 0.322 | 0.268 | +0.053 [nan, nan] | 1.000 | False |
| small_strong: glad_lo - ds_onecoin_nf10_lo | ap_key | 0.319 | 0.322 | -0.002 [nan, nan] | 1.000 | False |
| small_strong: catd - crh | ap_key | 0.210 | 0.209 | +0.001 [nan, nan] | 1.000 | False |
| small_strong: crh - mv | ap_key | 0.209 | 0.209 | +0.000 [nan, nan] | 1.000 | False |
| small_strong: entropy - mv | ap_key | 0.016 | 0.209 | -0.193 [nan, nan] | 0.165 | False |
| small_strong: crowdfm - ds_onecoin_nf10_lo | ap_key | 0.040 | 0.322 | -0.282 [nan, nan] | 0.696 | False |
| small_strong: ensemble - ds_onecoin_nf10_lo | ap_key | 0.372 | 0.322 | +0.050 [nan, nan] | 1.000 | False |
| small_strong: ensemble - crh | ap_key | 0.372 | 0.209 | +0.163 [nan, nan] | 0.320 | False |
| small_strong: ensemble - irt4pl | ap_key | 0.372 | 0.268 | +0.104 [nan, nan] | 1.000 | False |
| small_strong: accu_depen_lo - accu_nodep_lo | ap_key | 0.272 | 0.322 | -0.049 [nan, nan] | 1.000 | False |
| small_strong: ds_onecoin_nf10_lo - ds_onecoin_lo | ap_key | 0.322 | 0.083 | +0.239 [nan, nan] | 0.660 | False |
| small_strong: irt4pl_miss - irt4pl | ap_key | 0.416 | 0.268 | +0.148 [nan, nan] | 1.000 | False |
| offbench_strong: ds_onecoin_nf10_lo - crh | ap_key | 0.513 | 0.328 | +0.185 [nan, nan] | 1.000 | False |
| offbench_strong: ds_onecoin_nf10_lo - irt4pl | ap_key | 0.513 | 0.068 | +0.445 [nan, nan] | 0.102 | False |
| offbench_strong: glad_lo - ds_onecoin_nf10_lo | ap_key | 0.496 | 0.513 | -0.017 [nan, nan] | 1.000 | False |
| offbench_strong: catd - crh | ap_key | 0.321 | 0.328 | -0.007 [nan, nan] | 1.000 | False |
| offbench_strong: crh - mv | ap_key | 0.328 | 0.327 | +0.000 [nan, nan] | 1.000 | False |
| offbench_strong: entropy - mv | ap_key | 0.011 | 0.327 | -0.317 [nan, nan] | 0.035 | False |
| offbench_strong: crowdfm - ds_onecoin_nf10_lo | ap_key | 0.040 | 0.513 | -0.473 [nan, nan] | 0.019 | False |
| offbench_strong: ensemble - ds_onecoin_nf10_lo | ap_key | 0.461 | 0.513 | -0.052 [nan, nan] | 1.000 | False |
| offbench_strong: ensemble - crh | ap_key | 0.461 | 0.328 | +0.133 [nan, nan] | 1.000 | False |
| offbench_strong: ensemble - irt4pl | ap_key | 0.461 | 0.068 | +0.392 [nan, nan] | 0.019 | False |
| offbench_strong: accu_depen_lo - accu_nodep_lo | ap_key | 0.397 | 0.513 | -0.116 [nan, nan] | 0.019 | False |
| offbench_strong: ds_onecoin_nf10_lo - ds_onecoin_lo | ap_key | 0.513 | 0.218 | +0.295 [nan, nan] | 0.812 | False |
| offbench_strong: irt4pl_miss - irt4pl | ap_key | 0.044 | 0.068 | -0.024 [nan, nan] | 1.000 | False |

## RQ3-hellaswag

| contrast | metric | a | b | diff [95% CI] | Holm p | equiv |
|---|---|---|---|---|---|---|
| full: ds_onecoin_lo - crh | ap | 0.092 | 0.090 | +0.003 [-0.006, 0.013] | 1.000 | True |
| full: ds_onecoin_lo - irt4pl | ap | 0.092 | 0.085 | +0.007 [-0.007, 0.023] | 1.000 | False |
| full: glad_lo - ds_onecoin_lo | ap | 0.093 | 0.092 | +0.001 [-0.006, 0.006] | 1.000 | True |
| full: catd - crh | ap | 0.093 | 0.090 | +0.004 [0.002, 0.006] | 0.018 | True |
| full: crh - mv | ap | 0.090 | 0.089 | +0.001 [0.000, 0.003] | 0.018 | True |
| full: entropy - mv | ap | 0.055 | 0.089 | -0.034 [-0.052, -0.018] | 0.018 | False |
| full: crowdfm - ds_onecoin_lo | ap | 0.047 | 0.092 | -0.045 [-0.061, -0.032] | 0.018 | False |
| full: ensemble - ds_onecoin_lo | ap | 0.106 | 0.092 | +0.014 [0.005, 0.023] | 0.018 | False |
| full: ensemble - crh | ap | 0.106 | 0.090 | +0.016 [0.004, 0.031] | 0.208 | False |
| full: ensemble - irt4pl | ap | 0.106 | 0.085 | +0.021 [0.005, 0.040] | 0.264 | False |
| full: ds_full_lo - ds_onecoin_lo | ap | 0.090 | 0.092 | -0.002 [-0.010, 0.005] | 1.000 | True |
| small_strong: ds_onecoin_lo - crh | ap | 0.101 | 0.101 | -0.000 [-0.006, 0.005] | 1.000 | True |
| small_strong: ds_onecoin_lo - irt4pl | ap | 0.101 | 0.078 | +0.023 [0.009, 0.037] | 0.112 | False |
| small_strong: glad_lo - ds_onecoin_lo | ap | 0.101 | 0.101 | -0.000 [-0.001, 0.001] | 1.000 | True |
| small_strong: catd - crh | ap | 0.101 | 0.101 | -0.000 [-0.001, 0.000] | 1.000 | True |
| small_strong: crh - mv | ap | 0.101 | 0.101 | -0.000 [-0.001, 0.000] | 1.000 | True |
| small_strong: entropy - mv | ap | 0.069 | 0.101 | -0.033 [-0.050, -0.018] | 0.018 | False |
| small_strong: crowdfm - ds_onecoin_lo | ap | 0.085 | 0.101 | -0.016 [-0.026, -0.005] | 0.189 | False |
| small_strong: ensemble - ds_onecoin_lo | ap | 0.099 | 0.101 | -0.002 [-0.007, 0.003] | 1.000 | True |
| small_strong: ensemble - crh | ap | 0.099 | 0.101 | -0.002 [-0.007, 0.005] | 1.000 | True |
| small_strong: ensemble - irt4pl | ap | 0.099 | 0.078 | +0.021 [0.010, 0.033] | 0.058 | False |
| small_strong: accu_depen_lo - accu_nodep_lo | ap | 0.098 | 0.101 | -0.003 [-0.015, 0.011] | 1.000 | True |
| small_strong: ds_full_lo - ds_onecoin_lo | ap | 0.093 | 0.101 | -0.008 [-0.017, 0.002] | 1.000 | True |
| offbench_strong: ds_onecoin_lo - crh | ap | 0.091 | 0.089 | +0.001 [-0.002, 0.006] | 1.000 | True |
| offbench_strong: ds_onecoin_lo - irt4pl | ap | 0.091 | 0.090 | +0.001 [-0.014, 0.011] | 1.000 | True |
| offbench_strong: glad_lo - ds_onecoin_lo | ap | 0.091 | 0.091 | +0.000 [-0.000, 0.000] | 1.000 | True |
| offbench_strong: catd - crh | ap | 0.089 | 0.089 | -0.000 [-0.000, 0.000] | 1.000 | True |
| offbench_strong: crh - mv | ap | 0.089 | 0.089 | +0.000 [-0.000, 0.000] | 1.000 | True |
| offbench_strong: entropy - mv | ap | 0.066 | 0.089 | -0.023 [-0.040, -0.005] | 0.250 | False |
| offbench_strong: crowdfm - ds_onecoin_lo | ap | 0.081 | 0.091 | -0.009 [-0.016, -0.001] | 0.667 | True |
| offbench_strong: ensemble - ds_onecoin_lo | ap | 0.092 | 0.091 | +0.002 [-0.002, 0.005] | 1.000 | True |
| offbench_strong: ensemble - crh | ap | 0.092 | 0.089 | +0.003 [-0.001, 0.007] | 1.000 | True |
| offbench_strong: ensemble - irt4pl | ap | 0.092 | 0.090 | +0.002 [-0.011, 0.011] | 1.000 | True |
| offbench_strong: accu_depen_lo - accu_nodep_lo | ap | 0.082 | 0.091 | -0.009 [-0.017, 0.001] | 1.000 | True |
| offbench_strong: ds_full_lo - ds_onecoin_lo | ap | 0.089 | 0.091 | -0.002 [-0.007, 0.004] | 1.000 | True |
| full: ds_onecoin_lo - crh | ap_key | 0.129 | 0.123 | +0.006 [-0.005, 0.017] | 1.000 | True |
| full: ds_onecoin_lo - irt4pl | ap_key | 0.129 | 0.129 | -0.000 [-0.025, 0.022] | 1.000 | False |
| full: glad_lo - ds_onecoin_lo | ap_key | 0.129 | 0.129 | +0.000 [-0.007, 0.006] | 1.000 | True |
| full: catd - crh | ap_key | 0.129 | 0.123 | +0.006 [0.004, 0.009] | 0.018 | True |
| full: crh - mv | ap_key | 0.123 | 0.121 | +0.002 [0.001, 0.004] | 0.018 | True |
| full: entropy - mv | ap_key | 0.072 | 0.121 | -0.049 [-0.072, -0.027] | 0.018 | False |
| full: crowdfm - ds_onecoin_lo | ap_key | 0.061 | 0.129 | -0.068 [-0.089, -0.049] | 0.018 | False |
| full: ensemble - ds_onecoin_lo | ap_key | 0.147 | 0.129 | +0.018 [0.010, 0.029] | 0.018 | False |
| full: ensemble - crh | ap_key | 0.147 | 0.123 | +0.024 [0.010, 0.039] | 0.018 | False |
| full: ensemble - irt4pl | ap_key | 0.147 | 0.129 | +0.018 [-0.005, 0.040] | 1.000 | False |
| full: ds_full_lo - ds_onecoin_lo | ap_key | 0.124 | 0.129 | -0.005 [-0.015, 0.004] | 1.000 | True |
| small_strong: ds_onecoin_lo - crh | ap_key | 0.147 | 0.146 | +0.001 [-0.007, 0.009] | 1.000 | True |
| small_strong: ds_onecoin_lo - irt4pl | ap_key | 0.147 | 0.109 | +0.038 [0.018, 0.058] | 0.018 | False |
| small_strong: glad_lo - ds_onecoin_lo | ap_key | 0.146 | 0.147 | -0.000 [-0.002, 0.001] | 1.000 | True |
| small_strong: catd - crh | ap_key | 0.146 | 0.146 | -0.000 [-0.001, 0.001] | 1.000 | True |
| small_strong: crh - mv | ap_key | 0.146 | 0.147 | -0.000 [-0.001, 0.000] | 1.000 | True |
| small_strong: entropy - mv | ap_key | 0.094 | 0.147 | -0.052 [-0.076, -0.030] | 0.018 | False |
| small_strong: crowdfm - ds_onecoin_lo | ap_key | 0.117 | 0.147 | -0.029 [-0.044, -0.015] | 0.018 | False |
| small_strong: ensemble - ds_onecoin_lo | ap_key | 0.144 | 0.147 | -0.002 [-0.010, 0.005] | 1.000 | True |
| small_strong: ensemble - crh | ap_key | 0.144 | 0.146 | -0.002 [-0.010, 0.007] | 1.000 | True |
| small_strong: ensemble - irt4pl | ap_key | 0.144 | 0.109 | +0.036 [0.020, 0.051] | 0.018 | False |
| small_strong: accu_depen_lo - accu_nodep_lo | ap_key | 0.141 | 0.147 | -0.006 [-0.022, 0.013] | 1.000 | True |
| small_strong: ds_full_lo - ds_onecoin_lo | ap_key | 0.131 | 0.147 | -0.015 [-0.027, -0.002] | 0.851 | False |
| offbench_strong: ds_onecoin_lo - crh | ap_key | 0.127 | 0.126 | +0.001 [-0.005, 0.006] | 1.000 | True |
| offbench_strong: ds_onecoin_lo - irt4pl | ap_key | 0.127 | 0.127 | -0.001 [-0.019, 0.013] | 1.000 | True |
| offbench_strong: glad_lo - ds_onecoin_lo | ap_key | 0.126 | 0.127 | -0.000 [-0.001, 0.001] | 1.000 | True |
| offbench_strong: catd - crh | ap_key | 0.126 | 0.126 | -0.000 [-0.001, 0.000] | 1.000 | True |
| offbench_strong: crh - mv | ap_key | 0.126 | 0.126 | +0.000 [-0.000, 0.000] | 1.000 | True |
| offbench_strong: entropy - mv | ap_key | 0.087 | 0.126 | -0.039 [-0.061, -0.016] | 0.025 | False |
| offbench_strong: crowdfm - ds_onecoin_lo | ap_key | 0.111 | 0.127 | -0.016 [-0.026, -0.005] | 0.144 | False |
| offbench_strong: ensemble - ds_onecoin_lo | ap_key | 0.131 | 0.127 | +0.004 [-0.000, 0.008] | 1.000 | True |
| offbench_strong: ensemble - crh | ap_key | 0.131 | 0.126 | +0.005 [-0.001, 0.011] | 1.000 | True |
| offbench_strong: ensemble - irt4pl | ap_key | 0.131 | 0.127 | +0.003 [-0.013, 0.015] | 1.000 | True |
| offbench_strong: accu_depen_lo - accu_nodep_lo | ap_key | 0.118 | 0.127 | -0.009 [-0.021, 0.006] | 1.000 | True |
| offbench_strong: ds_full_lo - ds_onecoin_lo | ap_key | 0.125 | 0.127 | -0.001 [-0.008, 0.007] | 1.000 | True |

## RQ3-mmlu

| contrast | metric | a | b | diff [95% CI] | Holm p | equiv |
|---|---|---|---|---|---|---|
| full: ds_onecoin_lo - crh | ap | 0.190 | 0.145 | +0.045 [0.017, 0.075] | 0.019 | False |
| full: ds_onecoin_lo - irt4pl | ap | 0.190 | 0.119 | +0.072 [0.039, 0.109] | 0.018 | False |
| full: glad_lo - ds_onecoin_lo | ap | 0.191 | 0.190 | +0.001 [-0.012, 0.012] | 1.000 | True |
| full: catd - crh | ap | 0.162 | 0.145 | +0.016 [0.009, 0.026] | 0.018 | False |
| full: crh - mv | ap | 0.145 | 0.141 | +0.004 [0.002, 0.006] | 0.018 | True |
| full: entropy - mv | ap | 0.073 | 0.141 | -0.068 [-0.099, -0.044] | 0.018 | False |
| full: crowdfm - ds_onecoin_lo | ap | 0.099 | 0.190 | -0.091 [-0.131, -0.055] | 0.018 | False |
| full: ensemble - ds_onecoin_lo | ap | 0.208 | 0.190 | +0.017 [-0.002, 0.036] | 0.760 | False |
| full: ensemble - crh | ap | 0.208 | 0.145 | +0.063 [0.032, 0.093] | 0.018 | False |
| full: ensemble - irt4pl | ap | 0.208 | 0.119 | +0.089 [0.056, 0.126] | 0.018 | False |
| full: ds_full_lo - ds_onecoin_lo | ap | 0.175 | 0.190 | -0.015 [-0.033, 0.001] | 0.760 | False |
| small_strong: ds_onecoin_lo - crh | ap | 0.223 | 0.225 | -0.002 [-0.026, 0.023] | 1.000 | False |
| small_strong: ds_onecoin_lo - irt4pl | ap | 0.223 | 0.191 | +0.032 [0.003, 0.063] | 0.396 | False |
| small_strong: glad_lo - ds_onecoin_lo | ap | 0.219 | 0.223 | -0.004 [-0.009, 0.000] | 0.561 | True |
| small_strong: catd - crh | ap | 0.225 | 0.225 | -0.000 [-0.002, 0.001] | 1.000 | True |
| small_strong: crh - mv | ap | 0.225 | 0.221 | +0.004 [0.001, 0.008] | 0.019 | True |
| small_strong: entropy - mv | ap | 0.083 | 0.221 | -0.138 [-0.181, -0.097] | 0.018 | False |
| small_strong: crowdfm - ds_onecoin_lo | ap | 0.185 | 0.223 | -0.038 [-0.064, -0.011] | 0.019 | False |
| small_strong: ensemble - ds_onecoin_lo | ap | 0.232 | 0.223 | +0.009 [-0.008, 0.025] | 1.000 | False |
| small_strong: ensemble - crh | ap | 0.232 | 0.225 | +0.006 [-0.010, 0.024] | 1.000 | False |
| small_strong: ensemble - irt4pl | ap | 0.232 | 0.191 | +0.040 [0.020, 0.062] | 0.018 | False |
| small_strong: accu_depen_lo - accu_nodep_lo | ap | 0.222 | 0.223 | -0.001 [-0.019, 0.020] | 1.000 | True |
| small_strong: ds_full_lo - ds_onecoin_lo | ap | 0.197 | 0.223 | -0.026 [-0.042, -0.011] | 0.018 | False |
| offbench_strong: ds_onecoin_lo - crh | ap | 0.184 | 0.177 | +0.008 [-0.014, 0.030] | 1.000 | False |
| offbench_strong: ds_onecoin_lo - irt4pl | ap | 0.184 | 0.142 | +0.042 [0.007, 0.077] | 0.208 | False |
| offbench_strong: glad_lo - ds_onecoin_lo | ap | 0.179 | 0.184 | -0.005 [-0.010, -0.001] | 0.112 | True |
| offbench_strong: catd - crh | ap | 0.175 | 0.177 | -0.002 [-0.004, -0.001] | 0.018 | True |
| offbench_strong: crh - mv | ap | 0.177 | 0.178 | -0.001 [-0.002, -0.001] | 0.032 | True |
| offbench_strong: entropy - mv | ap | 0.082 | 0.178 | -0.097 [-0.132, -0.063] | 0.018 | False |
| offbench_strong: crowdfm - ds_onecoin_lo | ap | 0.133 | 0.184 | -0.052 [-0.080, -0.023] | 0.018 | False |
| offbench_strong: ensemble - ds_onecoin_lo | ap | 0.209 | 0.184 | +0.025 [0.015, 0.037] | 0.018 | False |
| offbench_strong: ensemble - crh | ap | 0.209 | 0.177 | +0.033 [0.016, 0.052] | 0.018 | False |
| offbench_strong: ensemble - irt4pl | ap | 0.209 | 0.142 | +0.068 [0.035, 0.100] | 0.018 | False |
| offbench_strong: accu_depen_lo - accu_nodep_lo | ap | 0.179 | 0.184 | -0.005 [-0.023, 0.015] | 1.000 | True |
| offbench_strong: ds_full_lo - ds_onecoin_lo | ap | 0.145 | 0.184 | -0.039 [-0.065, -0.014] | 0.032 | False |
| full: ds_onecoin_lo - crh | ap_key | 0.246 | 0.142 | +0.104 [0.032, 0.184] | 0.068 | False |
| full: ds_onecoin_lo - irt4pl | ap_key | 0.246 | 0.086 | +0.161 [0.081, 0.254] | 0.018 | False |
| full: glad_lo - ds_onecoin_lo | ap_key | 0.256 | 0.246 | +0.010 [-0.027, 0.042] | 1.000 | False |
| full: catd - crh | ap_key | 0.180 | 0.142 | +0.038 [0.019, 0.061] | 0.018 | False |
| full: crh - mv | ap_key | 0.142 | 0.134 | +0.008 [0.004, 0.016] | 0.018 | True |
| full: entropy - mv | ap_key | 0.021 | 0.134 | -0.113 [-0.171, -0.072] | 0.018 | False |
| full: crowdfm - ds_onecoin_lo | ap_key | 0.080 | 0.246 | -0.166 [-0.262, -0.084] | 0.018 | False |
| full: ensemble - ds_onecoin_lo | ap_key | 0.284 | 0.246 | +0.038 [-0.013, 0.084] | 1.000 | False |
| full: ensemble - crh | ap_key | 0.284 | 0.142 | +0.142 [0.065, 0.228] | 0.038 | False |
| full: ensemble - irt4pl | ap_key | 0.284 | 0.086 | +0.199 [0.116, 0.290] | 0.018 | False |
| full: ds_full_lo - ds_onecoin_lo | ap_key | 0.192 | 0.246 | -0.054 [-0.105, -0.013] | 0.140 | False |
| small_strong: ds_onecoin_lo - crh | ap_key | 0.313 | 0.315 | -0.001 [-0.066, 0.061] | 1.000 | False |
| small_strong: ds_onecoin_lo - irt4pl | ap_key | 0.313 | 0.214 | +0.100 [0.033, 0.164] | 0.068 | False |
| small_strong: glad_lo - ds_onecoin_lo | ap_key | 0.303 | 0.313 | -0.010 [-0.021, 0.001] | 0.737 | True |
| small_strong: catd - crh | ap_key | 0.314 | 0.315 | -0.000 [-0.003, 0.002] | 1.000 | True |
| small_strong: crh - mv | ap_key | 0.315 | 0.302 | +0.012 [0.004, 0.022] | 0.054 | False |
| small_strong: entropy - mv | ap_key | 0.026 | 0.302 | -0.276 [-0.366, -0.199] | 0.018 | False |
| small_strong: crowdfm - ds_onecoin_lo | ap_key | 0.225 | 0.313 | -0.088 [-0.164, -0.013] | 0.195 | False |
| small_strong: ensemble - ds_onecoin_lo | ap_key | 0.321 | 0.313 | +0.008 [-0.029, 0.044] | 1.000 | False |
| small_strong: ensemble - crh | ap_key | 0.321 | 0.315 | +0.006 [-0.035, 0.051] | 1.000 | False |
| small_strong: ensemble - irt4pl | ap_key | 0.321 | 0.214 | +0.108 [0.058, 0.158] | 0.018 | False |
| small_strong: accu_depen_lo - accu_nodep_lo | ap_key | 0.300 | 0.313 | -0.014 [-0.057, 0.033] | 1.000 | False |
| small_strong: ds_full_lo - ds_onecoin_lo | ap_key | 0.240 | 0.313 | -0.073 [-0.128, -0.030] | 0.018 | False |
| offbench_strong: ds_onecoin_lo - crh | ap_key | 0.230 | 0.205 | +0.025 [-0.023, 0.077] | 1.000 | False |
| offbench_strong: ds_onecoin_lo - irt4pl | ap_key | 0.230 | 0.109 | +0.121 [0.054, 0.193] | 0.018 | False |
| offbench_strong: glad_lo - ds_onecoin_lo | ap_key | 0.217 | 0.230 | -0.013 [-0.021, -0.006] | 0.018 | True |
| offbench_strong: catd - crh | ap_key | 0.203 | 0.205 | -0.001 [-0.004, 0.000] | 0.737 | True |
| offbench_strong: crh - mv | ap_key | 0.205 | 0.206 | -0.001 [-0.002, -0.000] | 0.384 | True |
| offbench_strong: entropy - mv | ap_key | 0.023 | 0.206 | -0.182 [-0.241, -0.128] | 0.018 | False |
| offbench_strong: crowdfm - ds_onecoin_lo | ap_key | 0.116 | 0.230 | -0.114 [-0.181, -0.043] | 0.020 | False |
| offbench_strong: ensemble - ds_onecoin_lo | ap_key | 0.274 | 0.230 | +0.044 [0.027, 0.062] | 0.018 | False |
| offbench_strong: ensemble - crh | ap_key | 0.274 | 0.205 | +0.069 [0.028, 0.118] | 0.018 | False |
| offbench_strong: ensemble - irt4pl | ap_key | 0.274 | 0.109 | +0.165 [0.097, 0.242] | 0.018 | False |
| offbench_strong: accu_depen_lo - accu_nodep_lo | ap_key | 0.207 | 0.230 | -0.023 [-0.064, 0.016] | 1.000 | False |
| offbench_strong: ds_full_lo - ds_onecoin_lo | ap_key | 0.139 | 0.230 | -0.091 [-0.157, -0.028] | 0.068 | False |

## diag

| contrast | metric | a | b | diff [95% CI] | Holm p | equiv |
|---|---|---|---|---|---|---|
| full: ds_onecoin_lo log-odds - probability | ap | 0.190 | 0.095 | +0.095 [0.061, 0.135] | 0.002 | False |
| full: ds_onecoin_lo log-odds - probability | ap_key | 0.246 | 0.049 | +0.197 [0.123, 0.292] | 0.002 | False |
| full: ds_onecoin_nf10_lo log-odds - probability | ap | 0.238 | 0.182 | +0.056 [0.021, 0.107] | 0.002 | False |
| full: ds_onecoin_nf10_lo log-odds - probability | ap_key | 0.219 | 0.069 | +0.150 [nan, nan] | 0.065 | False |
| full: ds_onecoin_lo log-odds - probability | ap | 0.092 | 0.074 | +0.018 [0.010, 0.032] | 0.002 | False |
| full: ds_onecoin_lo log-odds - probability | ap_key | 0.129 | 0.099 | +0.029 [0.016, 0.046] | 0.002 | False |

## MMLU subject-cluster bootstrap

| contrast | metric | diff [95% CI] | p_boot |
|---|---|---|---|
| mv: strong - full | ap | +0.080 [0.027, 0.128] | 0.0005 |
| mv: offbench - full | ap | +0.037 [0.009, 0.067] | 0.0040 |
| mv: offbench - strong | ap | -0.043 [-0.079, -0.011] | 0.0005 |
| crh: strong - full | ap | +0.080 [0.027, 0.129] | 0.0005 |
| crh: offbench - full | ap | +0.032 [0.005, 0.059] | 0.0140 |
| crh: offbench - strong | ap | -0.049 [-0.086, -0.013] | 0.0005 |
| ds_onecoin_lo: strong - full | ap | +0.032 [-0.003, 0.067] | 0.0790 |
| ds_onecoin_lo: offbench - full | ap | -0.006 [-0.029, 0.014] | 0.4870 |
| ds_onecoin_lo: offbench - strong | ap | -0.039 [-0.067, -0.014] | 0.0010 |
| glad_lo: strong - full | ap | +0.027 [-0.008, 0.066] | 0.1530 |
| glad_lo: offbench - full | ap | -0.012 [-0.034, 0.006] | 0.2250 |
| glad_lo: offbench - strong | ap | -0.040 [-0.071, -0.012] | 0.0050 |
| irt4pl: strong - full | ap | +0.073 [0.022, 0.131] | 0.0005 |
| irt4pl: offbench - full | ap | +0.023 [0.002, 0.047] | 0.0340 |
| irt4pl: offbench - strong | ap | -0.049 [-0.093, -0.008] | 0.0070 |
| crh: top10 - random10 | ap | +0.068 [0.028, 0.118] | 0.0005 |
| crh: top30 - random30 | ap | +0.045 [0.015, 0.081] | 0.0005 |
| crh: top100 - random100 | ap | +0.039 [0.010, 0.069] | 0.0020 |
| crh: top300 - random300 | ap | +0.040 [0.010, 0.073] | 0.0020 |
| crh: top1000 - random1000 | ap | +0.061 [0.015, 0.105] | 0.0010 |
| crh: random300 - all | ap | -0.001 [-0.010, 0.004] | 0.8240 |
| ds_onecoin_lo: top10 - random10 | ap | +0.066 [0.021, 0.120] | 0.0005 |
| ds_onecoin_lo: top30 - random30 | ap | +0.022 [-0.004, 0.052] | 0.1270 |
| ds_onecoin_lo: top100 - random100 | ap | +0.002 [-0.020, 0.024] | 0.9750 |
| ds_onecoin_lo: top300 - random300 | ap | +0.002 [-0.021, 0.023] | 0.9170 |
| ds_onecoin_lo: top1000 - random1000 | ap | +0.018 [-0.003, 0.036] | 0.0980 |
| ds_onecoin_lo: random300 - all | ap | -0.002 [-0.005, 0.002] | 0.3130 |
| glad_lo: top10 - random10 | ap | +0.063 [0.020, 0.116] | 0.0005 |
| glad_lo: top30 - random30 | ap | +0.016 [-0.010, 0.045] | 0.3040 |
| glad_lo: top100 - random100 | ap | -0.002 [-0.024, 0.019] | 0.7920 |
| glad_lo: top300 - random300 | ap | -0.005 [-0.025, 0.014] | 0.5340 |
| glad_lo: top1000 - random1000 | ap | +0.007 [-0.009, 0.022] | 0.4380 |
| glad_lo: random300 - all | ap | -0.002 [-0.007, 0.003] | 0.4380 |
| full: ds_onecoin_lo - crh | ap | +0.045 [0.014, 0.077] | 0.0010 |
| full: ds_onecoin_lo - irt4pl | ap | +0.072 [0.021, 0.129] | 0.0005 |
| full: glad_lo - ds_onecoin_lo | ap | +0.001 [-0.013, 0.013] | 0.9840 |
| full: catd - crh | ap | +0.016 [0.002, 0.033] | 0.0050 |
| full: crh - mv | ap | +0.004 [0.001, 0.008] | 0.0005 |
| full: entropy - mv | ap | -0.068 [-0.154, -0.016] | 0.0005 |
| full: crowdfm - ds_onecoin_lo | ap | -0.091 [-0.160, -0.035] | 0.0005 |
| full: ensemble - ds_onecoin_lo | ap | +0.017 [-0.011, 0.043] | 0.3240 |
| full: ensemble - crh | ap | +0.063 [0.015, 0.110] | 0.0020 |
| full: ensemble - irt4pl | ap | +0.089 [0.022, 0.166] | 0.0010 |
| full: ds_full_lo - ds_onecoin_lo | ap | -0.015 [-0.035, 0.006] | 0.1920 |
| small_strong: ds_onecoin_lo - crh | ap | -0.002 [-0.035, 0.028] | 0.9710 |
| small_strong: ds_onecoin_lo - irt4pl | ap | +0.032 [-0.012, 0.079] | 0.1740 |
| small_strong: glad_lo - ds_onecoin_lo | ap | -0.004 [-0.010, 0.001] | 0.0760 |
| small_strong: catd - crh | ap | -0.000 [-0.001, 0.001] | 0.6280 |
| small_strong: crh - mv | ap | +0.004 [0.001, 0.008] | 0.0130 |
| small_strong: entropy - mv | ap | -0.138 [-0.252, -0.047] | 0.0005 |
| small_strong: crowdfm - ds_onecoin_lo | ap | -0.038 [-0.064, -0.013] | 0.0005 |
| small_strong: ensemble - ds_onecoin_lo | ap | +0.009 [-0.011, 0.030] | 0.4410 |
| small_strong: ensemble - crh | ap | +0.006 [-0.011, 0.025] | 0.4350 |
| small_strong: ensemble - irt4pl | ap | +0.040 [0.010, 0.074] | 0.0040 |
| small_strong: accu_depen_lo - accu_nodep_lo | ap | -0.001 [-0.018, 0.018] | 0.9960 |
| small_strong: ds_full_lo - ds_onecoin_lo | ap | -0.026 [-0.046, -0.006] | 0.0040 |
| offbench_strong: ds_onecoin_lo - crh | ap | +0.008 [-0.014, 0.030] | 0.5530 |
| offbench_strong: ds_onecoin_lo - irt4pl | ap | +0.042 [-0.005, 0.093] | 0.0980 |
| offbench_strong: glad_lo - ds_onecoin_lo | ap | -0.005 [-0.014, 0.001] | 0.1160 |
| offbench_strong: catd - crh | ap | -0.002 [-0.004, -0.001] | 0.0040 |
| offbench_strong: crh - mv | ap | -0.001 [-0.003, -0.000] | 0.0120 |
| offbench_strong: entropy - mv | ap | -0.097 [-0.202, -0.023] | 0.0005 |
| offbench_strong: crowdfm - ds_onecoin_lo | ap | -0.052 [-0.095, -0.012] | 0.0005 |
| offbench_strong: ensemble - ds_onecoin_lo | ap | +0.025 [0.008, 0.045] | 0.0005 |
| offbench_strong: ensemble - crh | ap | +0.033 [0.008, 0.059] | 0.0020 |
| offbench_strong: ensemble - irt4pl | ap | +0.068 [0.012, 0.132] | 0.0090 |
| offbench_strong: accu_depen_lo - accu_nodep_lo | ap | -0.005 [-0.025, 0.017] | 0.7880 |
| offbench_strong: ds_full_lo - ds_onecoin_lo | ap | -0.039 [-0.083, 0.003] | 0.0770 |
| full: ds_onecoin_lo log-odds - probability | ap | +0.095 [0.036, 0.170] | 0.0005 |
| mv: strong - full | ap_key | +0.169 [0.086, 0.221] | 0.0005 |
| mv: offbench - full | ap_key | +0.072 [0.017, 0.119] | 0.0120 |
| mv: offbench - strong | ap_key | -0.097 [-0.147, -0.039] | 0.0005 |
| crh: strong - full | ap_key | +0.173 [0.091, 0.224] | 0.0005 |
| crh: offbench - full | ap_key | +0.063 [0.006, 0.105] | 0.0280 |
| crh: offbench - strong | ap_key | -0.110 [-0.160, -0.052] | 0.0005 |
| ds_onecoin_lo: strong - full | ap_key | +0.067 [-0.015, 0.130] | 0.1070 |
| ds_onecoin_lo: offbench - full | ap_key | -0.016 [-0.079, 0.029] | 0.4690 |
| ds_onecoin_lo: offbench - strong | ap_key | -0.083 [-0.143, -0.019] | 0.0120 |
| glad_lo: strong - full | ap_key | +0.047 [-0.039, 0.119] | 0.3090 |
| glad_lo: offbench - full | ap_key | -0.039 [-0.101, 0.008] | 0.1080 |
| glad_lo: offbench - strong | ap_key | -0.086 [-0.150, -0.016] | 0.0100 |
| irt4pl: strong - full | ap_key | +0.128 [0.037, 0.212] | 0.0080 |
| irt4pl: offbench - full | ap_key | +0.024 [-0.025, 0.054] | 0.2590 |
| irt4pl: offbench - strong | ap_key | -0.104 [-0.166, -0.043] | 0.0005 |
| crh: top10 - random10 | ap_key | +0.100 [0.045, 0.170] | 0.0005 |
| crh: top30 - random30 | ap_key | +0.080 [0.028, 0.138] | 0.0005 |
| crh: top100 - random100 | ap_key | +0.080 [0.027, 0.130] | 0.0010 |
| crh: top300 - random300 | ap_key | +0.088 [0.031, 0.135] | 0.0040 |
| crh: top1000 - random1000 | ap_key | +0.159 [0.052, 0.227] | 0.0005 |
| crh: random300 - all | ap_key | -0.003 [-0.027, 0.009] | 0.7330 |
| ds_onecoin_lo: top10 - random10 | ap_key | +0.118 [0.040, 0.200] | 0.0005 |
| ds_onecoin_lo: top30 - random30 | ap_key | +0.053 [-0.004, 0.105] | 0.0730 |
| ds_onecoin_lo: top100 - random100 | ap_key | +0.003 [-0.059, 0.051] | 0.9760 |
| ds_onecoin_lo: top300 - random300 | ap_key | -0.002 [-0.062, 0.046] | 0.9240 |
| ds_onecoin_lo: top1000 - random1000 | ap_key | +0.037 [-0.015, 0.078] | 0.1320 |
| ds_onecoin_lo: random300 - all | ap_key | -0.003 [-0.013, 0.006] | 0.4660 |
| glad_lo: top10 - random10 | ap_key | +0.113 [0.035, 0.200] | 0.0005 |
| glad_lo: top30 - random30 | ap_key | +0.041 [-0.020, 0.096] | 0.2110 |
| glad_lo: top100 - random100 | ap_key | -0.006 [-0.063, 0.038] | 0.7410 |
| glad_lo: top300 - random300 | ap_key | -0.023 [-0.074, 0.024] | 0.3140 |
| glad_lo: top1000 - random1000 | ap_key | +0.009 [-0.030, 0.056] | 0.6600 |
| glad_lo: random300 - all | ap_key | -0.008 [-0.026, 0.006] | 0.2560 |
| full: ds_onecoin_lo - crh | ap_key | +0.104 [0.032, 0.176] | 0.0010 |
| full: ds_onecoin_lo - irt4pl | ap_key | +0.161 [0.046, 0.258] | 0.0070 |
| full: glad_lo - ds_onecoin_lo | ap_key | +0.010 [-0.026, 0.049] | 0.6470 |
| full: catd - crh | ap_key | +0.038 [0.004, 0.067] | 0.0180 |
| full: crh - mv | ap_key | +0.008 [0.001, 0.018] | 0.0005 |
| full: entropy - mv | ap_key | -0.113 [-0.248, -0.029] | 0.0010 |
| full: crowdfm - ds_onecoin_lo | ap_key | -0.166 [-0.260, -0.081] | 0.0005 |
| full: ensemble - ds_onecoin_lo | ap_key | +0.038 [-0.024, 0.088] | 0.2410 |
| full: ensemble - crh | ap_key | +0.142 [0.042, 0.224] | 0.0010 |
| full: ensemble - irt4pl | ap_key | +0.199 [0.060, 0.320] | 0.0010 |
| full: ds_full_lo - ds_onecoin_lo | ap_key | -0.054 [-0.102, 0.003] | 0.0620 |
| small_strong: ds_onecoin_lo - crh | ap_key | -0.001 [-0.068, 0.074] | 0.9560 |
| small_strong: ds_onecoin_lo - irt4pl | ap_key | +0.100 [-0.001, 0.190] | 0.0520 |
| small_strong: glad_lo - ds_onecoin_lo | ap_key | -0.010 [-0.021, 0.004] | 0.1370 |
| small_strong: catd - crh | ap_key | -0.000 [-0.002, 0.001] | 0.7020 |
| small_strong: crh - mv | ap_key | +0.012 [0.005, 0.020] | 0.0005 |
| small_strong: entropy - mv | ap_key | -0.276 [-0.427, -0.137] | 0.0005 |
| small_strong: crowdfm - ds_onecoin_lo | ap_key | -0.088 [-0.169, -0.041] | 0.0005 |
| small_strong: ensemble - ds_onecoin_lo | ap_key | +0.008 [-0.029, 0.042] | 0.6660 |
| small_strong: ensemble - crh | ap_key | +0.006 [-0.031, 0.060] | 0.7970 |
| small_strong: ensemble - irt4pl | ap_key | +0.108 [0.036, 0.169] | 0.0005 |
| small_strong: accu_depen_lo - accu_nodep_lo | ap_key | -0.014 [-0.059, 0.028] | 0.5600 |
| small_strong: ds_full_lo - ds_onecoin_lo | ap_key | -0.073 [-0.121, -0.019] | 0.0010 |
| offbench_strong: ds_onecoin_lo - crh | ap_key | +0.025 [-0.018, 0.077] | 0.3070 |
| offbench_strong: ds_onecoin_lo - irt4pl | ap_key | +0.121 [0.029, 0.209] | 0.0020 |
| offbench_strong: glad_lo - ds_onecoin_lo | ap_key | -0.013 [-0.026, -0.002] | 0.0060 |
| offbench_strong: catd - crh | ap_key | -0.001 [-0.004, 0.000] | 0.1500 |
| offbench_strong: crh - mv | ap_key | -0.001 [-0.002, 0.000] | 0.0530 |
| offbench_strong: entropy - mv | ap_key | -0.182 [-0.335, -0.068] | 0.0005 |
| offbench_strong: crowdfm - ds_onecoin_lo | ap_key | -0.114 [-0.177, -0.048] | 0.0010 |
| offbench_strong: ensemble - ds_onecoin_lo | ap_key | +0.044 [0.016, 0.069] | 0.0005 |
| offbench_strong: ensemble - crh | ap_key | +0.069 [0.019, 0.119] | 0.0040 |
| offbench_strong: ensemble - irt4pl | ap_key | +0.165 [0.056, 0.267] | 0.0005 |
| offbench_strong: accu_depen_lo - accu_nodep_lo | ap_key | -0.023 [-0.066, 0.023] | 0.3560 |
| offbench_strong: ds_full_lo - ds_onecoin_lo | ap_key | -0.091 [-0.168, 0.007] | 0.0740 |
| full: ds_onecoin_lo log-odds - probability | ap_key | +0.197 [0.081, 0.314] | 0.0005 |