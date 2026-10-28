#!/bin/bash
# Barcha bosqichlarni ketma-ket bajaradi
cd "$(dirname "$0")" || exit 1
echo "=== s1_1_qadam_bazaviy_model_va_fgsm_aniqliknin.py ==="; python3 s1_1_qadam_bazaviy_model_va_fgsm_aniqliknin.py || echo "XATO: s1_1_qadam_bazaviy_model_va_fgsm_aniqliknin.py"
echo "=== s2_2_qadam_pgd_hujumini_qolda_yozish_va_fgs.py ==="; python3 s2_2_qadam_pgd_hujumini_qolda_yozish_va_fgs.py || echo "XATO: s2_2_qadam_pgd_hujumini_qolda_yozish_va_fgs.py"
echo "=== s3_3_qadam_adversarial_oqitish_qisqa_va_mus.py ==="; python3 s3_3_qadam_adversarial_oqitish_qisqa_va_mus.py || echo "XATO: s3_3_qadam_adversarial_oqitish_qisqa_va_mus.py"
echo "=== s4_4_qadam_gradient_saliency_xaritasi_toza_.py ==="; python3 s4_4_qadam_gradient_saliency_xaritasi_toza_.py || echo "XATO: s4_4_qadam_gradient_saliency_xaritasi_toza_.py"
echo "=== s5_5_qadam_adolatlilik_metricframe_va_thres.py ==="; python3 s5_5_qadam_adolatlilik_metricframe_va_thres.py || echo "XATO: s5_5_qadam_adolatlilik_metricframe_va_thres.py"
echo "=== s6_6_qadam_shap_treeexplainer_global_muhiml.py ==="; python3 s6_6_qadam_shap_treeexplainer_global_muhiml.py || echo "XATO: s6_6_qadam_shap_treeexplainer_global_muhiml.py"
echo "=== s7_7_qadam_etik_tekshiruv_royxatini_avtomat.py ==="; python3 s7_7_qadam_etik_tekshiruv_royxatini_avtomat.py || echo "XATO: s7_7_qadam_etik_tekshiruv_royxatini_avtomat.py"
