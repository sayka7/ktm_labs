#!/bin/bash
# Barcha bosqichlarni ketma-ket bajaradi
cd "$(dirname "$0")" || exit 1
echo "=== s1_1_qadam_arxitekturalar_pasporti_va_roofl.py ==="; python3 s1_1_qadam_arxitekturalar_pasporti_va_roofl.py || echo "XATO: s1_1_qadam_arxitekturalar_pasporti_va_roofl.py"
echo "=== s2_2_qadam_tpu_sistolik_massivi_takt_daraja.py ==="; python3 s2_2_qadam_tpu_sistolik_massivi_takt_daraja.py || echo "XATO: s2_2_qadam_tpu_sistolik_massivi_takt_daraja.py"
echo "=== s3_3_qadam_fpga_quvur_pipeline_ii_va_fixed_.py ==="; python3 s3_3_qadam_fpga_quvur_pipeline_ii_va_fixed_.py || echo "XATO: s3_3_qadam_fpga_quvur_pipeline_ii_va_fixed_.py"
echo "=== s4_4_qadam_npu_uchun_int8_kvantlash_aniqlik.py ==="; python3 s4_4_qadam_npu_uchun_int8_kvantlash_aniqlik.py || echo "XATO: s4_4_qadam_npu_uchun_int8_kvantlash_aniqlik.py"
echo "=== s5_5_qadam_real_cpuda_gemm_va_konvolyutsiya.py ==="; python3 s5_5_qadam_real_cpuda_gemm_va_konvolyutsiya.py || echo "XATO: s5_5_qadam_real_cpuda_gemm_va_konvolyutsiya.py"
echo "=== s6_6_qadam_energiya_samaradorligi_top_s_vt_.py ==="; python3 s6_6_qadam_energiya_samaradorligi_top_s_vt_.py || echo "XATO: s6_6_qadam_energiya_samaradorligi_top_s_vt_.py"
echo "=== s7_7_qadam_yakuniy_tanlov_matritsasi_wsm_to.py ==="; python3 s7_7_qadam_yakuniy_tanlov_matritsasi_wsm_to.py || echo "XATO: s7_7_qadam_yakuniy_tanlov_matritsasi_wsm_to.py"
