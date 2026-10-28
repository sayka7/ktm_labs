#!/bin/bash
# Barcha bosqichlarni ketma-ket bajaradi
cd "$(dirname "$0")" || exit 1
echo "=== s1_1_qadam_ai_tizimi_xavf_reyestri_generato.py ==="; python3 s1_1_qadam_ai_tizimi_xavf_reyestri_generato.py || echo "XATO: s1_1_qadam_ai_tizimi_xavf_reyestri_generato.py"
echo "=== s2_2_qadam_eu_ai_act_xavf_toifasini_aniqlov.py ==="; python3 s2_2_qadam_eu_ai_act_xavf_toifasini_aniqlov.py || echo "XATO: s2_2_qadam_eu_ai_act_xavf_toifasini_aniqlov.py"
echo "=== s3_3_qadam_fairness_audit_kesishgan_guruhla.py ==="; python3 s3_3_qadam_fairness_audit_kesishgan_guruhla.py || echo "XATO: s3_3_qadam_fairness_audit_kesishgan_guruhla.py"
echo "=== s4_4_qadam_mustahkamlik_testi_shovqin_senso.py ==="; python3 s4_4_qadam_mustahkamlik_testi_shovqin_senso.py || echo "XATO: s4_4_qadam_mustahkamlik_testi_shovqin_senso.py"
echo "=== s5_5_qadam_tushuntiriluvchanlik_permutatsio.py ==="; python3 s5_5_qadam_tushuntiriluvchanlik_permutatsio.py || echo "XATO: s5_5_qadam_tushuntiriluvchanlik_permutatsio.py"
echo "=== s6_6_qadam_muvofiqlik_tekshiruv_royxati_va_.py ==="; python3 s6_6_qadam_muvofiqlik_tekshiruv_royxati_va_.py || echo "XATO: s6_6_qadam_muvofiqlik_tekshiruv_royxati_va_.py"
echo "=== s7_7_qadam_takomillashtirish_rejasi_byudjet.py ==="; python3 s7_7_qadam_takomillashtirish_rejasi_byudjet.py || echo "XATO: s7_7_qadam_takomillashtirish_rejasi_byudjet.py"
