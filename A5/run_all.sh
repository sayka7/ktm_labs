#!/bin/bash
# Barcha bosqichlarni ketma-ket bajaradi
cd "$(dirname "$0")" || exit 1
echo "=== s1_1_qadam_ddp_gloo_backend_bilan_2_jarayon.py ==="; python3 s1_1_qadam_ddp_gloo_backend_bilan_2_jarayon.py || echo "XATO: s1_1_qadam_ddp_gloo_backend_bilan_2_jarayon.py"
echo "=== s2_2_qadam_masshtablash_1_vs_2_jarayon_s_n_.py ==="; python3 s2_2_qadam_masshtablash_1_vs_2_jarayon_s_n_.py || echo "XATO: s2_2_qadam_masshtablash_1_vs_2_jarayon_s_n_.py"
echo "=== s3_3_qadam_halqa_all_reduce_ni_numpyda_qold.py ==="; python3 s3_3_qadam_halqa_all_reduce_ni_numpyda_qold.py || echo "XATO: s3_3_qadam_halqa_all_reduce_ni_numpyda_qold.py"
echo "=== s4_4_qadam_gradient_toplash_va_malumotlar_p.py ==="; python3 s4_4_qadam_gradient_toplash_va_malumotlar_p.py || echo "XATO: s4_4_qadam_gradient_toplash_va_malumotlar_p.py"
echo "=== s5_5_qadam_parallel_inferens_sharding_ishch.py ==="; python3 s5_5_qadam_parallel_inferens_sharding_ishch.py || echo "XATO: s5_5_qadam_parallel_inferens_sharding_ishch.py"
echo "=== s6_6_qadam_amdal_va_gustafson_qonunlari_gra.py ==="; python3 s6_6_qadam_amdal_va_gustafson_qonunlari_gra.py || echo "XATO: s6_6_qadam_amdal_va_gustafson_qonunlari_gra.py"
echo "=== s7_7_qadam_aloqa_hisob_nisbati_kalkulyatori.py ==="; python3 s7_7_qadam_aloqa_hisob_nisbati_kalkulyatori.py || echo "XATO: s7_7_qadam_aloqa_hisob_nisbati_kalkulyatori.py"
