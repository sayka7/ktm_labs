#!/bin/bash
# Barcha bosqichlarni ketma-ket bajaradi
cd "$(dirname "$0")" || exit 1
echo "=== s1_1_qadam_etalon_model_va_olchov_metodikas.py ==="; python3 s1_1_qadam_etalon_model_va_olchov_metodikas.py || echo "XATO: s1_1_qadam_etalon_model_va_olchov_metodikas.py"
echo "=== s2_2_qadam_onnx_eksport_grafik_optimallasht.py ==="; python3 s2_2_qadam_onnx_eksport_grafik_optimallasht.py || echo "XATO: s2_2_qadam_onnx_eksport_grafik_optimallasht.py"
echo "=== s3_3_qadam_hisoblash_aniqligi_formatlari_fp.py ==="; python3 s3_3_qadam_hisoblash_aniqligi_formatlari_fp.py || echo "XATO: s3_3_qadam_hisoblash_aniqligi_formatlari_fp.py"
echo "=== s4_4_qadam_paket_va_oqim_tahlili_s_b_modeli.py ==="; python3 s4_4_qadam_paket_va_oqim_tahlili_s_b_modeli.py || echo "XATO: s4_4_qadam_paket_va_oqim_tahlili_s_b_modeli.py"
echo "=== s5_5_qadam_parallellik_intra_op_oqimlar_ses.py ==="; python3 s5_5_qadam_parallellik_intra_op_oqimlar_ses.py || echo "XATO: s5_5_qadam_parallellik_intra_op_oqimlar_ses.py"
echo "=== s6_6_qadam_energiya_samaradorligi_j_inferen.py ==="; python3 s6_6_qadam_energiya_samaradorligi_j_inferen.py || echo "XATO: s6_6_qadam_energiya_samaradorligi_j_inferen.py"
echo "=== s7_7_qadam_yakuniy_pareto_tahlili_ssenariy_.py ==="; python3 s7_7_qadam_yakuniy_pareto_tahlili_ssenariy_.py || echo "XATO: s7_7_qadam_yakuniy_pareto_tahlili_ssenariy_.py"
