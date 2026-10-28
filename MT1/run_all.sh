#!/bin/bash
# Barcha bosqichlarni ketma-ket bajaradi
cd "$(dirname "$0")" || exit 1
echo "=== s1_1_qadam_hayot_sikli_xaritasi_sintetik_re.py ==="; python3 s1_1_qadam_hayot_sikli_xaritasi_sintetik_re.py || echo "XATO: s1_1_qadam_hayot_sikli_xaritasi_sintetik_re.py"
echo "=== s2_2_qadam_tavsiya_modellari_item_knn_matri.py ==="; python3 s2_2_qadam_tavsiya_modellari_item_knn_matri.py || echo "XATO: s2_2_qadam_tavsiya_modellari_item_knn_matri.py"
echo "=== s3_3_qadam_onlayn_tajriba_a_b_test_statisti.py ==="; python3 s3_3_qadam_onlayn_tajriba_a_b_test_statisti.py || echo "XATO: s3_3_qadam_onlayn_tajriba_a_b_test_statisti.py"
echo "=== s4_4_qadam_uber_uslubidagi_misol_talab_prog.py ==="; python3 s4_4_qadam_uber_uslubidagi_misol_talab_prog.py || echo "XATO: s4_4_qadam_uber_uslubidagi_misol_talab_prog.py"
echo "=== s5_5_qadam_amazon_uslubidagi_ombor_abc_tahl.py ==="; python3 s5_5_qadam_amazon_uslubidagi_ombor_abc_tahl.py || echo "XATO: s5_5_qadam_amazon_uslubidagi_ombor_abc_tahl.py"
echo "=== s6_6_qadam_model_monitoringi_drift_psi_ks_h.py ==="; python3 s6_6_qadam_model_monitoringi_drift_psi_ks_h.py || echo "XATO: s6_6_qadam_model_monitoringi_drift_psi_ks_h.py"
echo "=== s7_7_qadam_joylash_va_hujjat_api_xizmat_kan.py ==="; python3 s7_7_qadam_joylash_va_hujjat_api_xizmat_kan.py || echo "XATO: s7_7_qadam_joylash_va_hujjat_api_xizmat_kan.py"
