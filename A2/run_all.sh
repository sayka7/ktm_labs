#!/bin/bash
# Barcha bosqichlarni ketma-ket bajaradi
cd "$(dirname "$0")" || exit 1
echo "=== s1_1_qadam_sof_python_vs_numpy_vektorlash.py ==="; python3 s1_1_qadam_sof_python_vs_numpy_vektorlash.py || echo "XATO: s1_1_qadam_sof_python_vs_numpy_vektorlash.py"
echo "=== s2_2_qadam_numba_njit_va_prange_openmpning_.py ==="; python3 s2_2_qadam_numba_njit_va_prange_openmpning_.py || echo "XATO: s2_2_qadam_numba_njit_va_prange_openmpning_.py"
echo "=== s3_3_qadam_kesh_bloklash_tiled_vs_naive_mat.py ==="; python3 s3_3_qadam_kesh_bloklash_tiled_vs_naive_mat.py || echo "XATO: s3_3_qadam_kesh_bloklash_tiled_vs_naive_mat.py"
echo "=== s4_4_qadam_konvolyutsiyani_tort_usulda_para.py ==="; python3 s4_4_qadam_konvolyutsiyani_tort_usulda_para.py || echo "XATO: s4_4_qadam_konvolyutsiyani_tort_usulda_para.py"
echo "=== s5_5_qadam_pytorch_gpu_orniga_kop_yadro_1_v.py ==="; python3 s5_5_qadam_pytorch_gpu_orniga_kop_yadro_1_v.py || echo "XATO: s5_5_qadam_pytorch_gpu_orniga_kop_yadro_1_v.py"
echo "=== s6_6_qadam_malumot_parallelligi_multiproces.py ==="; python3 s6_6_qadam_malumot_parallelligi_multiproces.py || echo "XATO: s6_6_qadam_malumot_parallelligi_multiproces.py"
echo "=== s7_7_qadam_amdahl_va_gustafson_qonunlari_na.py ==="; python3 s7_7_qadam_amdahl_va_gustafson_qonunlari_na.py || echo "XATO: s7_7_qadam_amdahl_va_gustafson_qonunlari_na.py"
