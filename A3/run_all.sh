#!/bin/bash
# Barcha bosqichlarni ketma-ket bajaradi
cd "$(dirname "$0")" || exit 1
echo "=== s1_1_qadam_etalon_pytorch_fp32_oqitish_va_o.py ==="; python3 s1_1_qadam_etalon_pytorch_fp32_oqitish_va_o.py || echo "XATO: s1_1_qadam_etalon_pytorch_fp32_oqitish_va_o.py"
echo "=== s2_2_qadam_onnx_runtime_optimallashtirish_d.py ==="; python3 s2_2_qadam_onnx_runtime_optimallashtirish_d.py || echo "XATO: s2_2_qadam_onnx_runtime_optimallashtirish_d.py"
echo "=== s3_3_qadam_pytorchda_int8_dinamik_kvantlash.py ==="; python3 s3_3_qadam_pytorchda_int8_dinamik_kvantlash.py || echo "XATO: s3_3_qadam_pytorchda_int8_dinamik_kvantlash.py"
echo "=== s4_4_qadam_onnx_runtimeda_kvantlash_dinamik.py ==="; python3 s4_4_qadam_onnx_runtimeda_kvantlash_dinamik.py || echo "XATO: s4_4_qadam_onnx_runtimeda_kvantlash_dinamik.py"
echo "=== s5_5_qadam_yakuniy_jadval_pareto_tahlili_va.py ==="; python3 s5_5_qadam_yakuniy_jadval_pareto_tahlili_va.py || echo "XATO: s5_5_qadam_yakuniy_jadval_pareto_tahlili_va.py"
echo "=== s6_6_qadam_energiya_bahosi_nusxalar_rejasi_.py ==="; python3 s6_6_qadam_energiya_bahosi_nusxalar_rejasi_.py || echo "XATO: s6_6_qadam_energiya_bahosi_nusxalar_rejasi_.py"
