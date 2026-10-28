#!/bin/bash
# Barcha bosqichlarni ketma-ket bajaradi
cd "$(dirname "$0")" || exit 1
echo "=== s1_1_qadam_fastapi_inferens_xizmati_probela.py ==="; python3 s1_1_qadam_fastapi_inferens_xizmati_probela.py || echo "XATO: s1_1_qadam_fastapi_inferens_xizmati_probela.py"
echo "=== s2_2_qadam_yuklama_sinovi_ketma_ket_va_para.py ==="; python3 s2_2_qadam_yuklama_sinovi_ketma_ket_va_para.py || echo "XATO: s2_2_qadam_yuklama_sinovi_ketma_ket_va_para.py"
echo "=== s3_3_qadam_dockerfile_tahlili_va_tasvir_haj.py ==="; python3 s3_3_qadam_dockerfile_tahlili_va_tasvir_haj.py || echo "XATO: s3_3_qadam_dockerfile_tahlili_va_tasvir_haj.py"
echo "=== s4_4_qadam_kubernetes_manifestlari_pyyaml_b.py ==="; python3 s4_4_qadam_kubernetes_manifestlari_pyyaml_b.py || echo "XATO: s4_4_qadam_kubernetes_manifestlari_pyyaml_b.py"
echo "=== s5_5_qadam_hpa_avtomasshtablash_simulyatsiy.py ==="; python3 s5_5_qadam_hpa_avtomasshtablash_simulyatsiy.py || echo "XATO: s5_5_qadam_hpa_avtomasshtablash_simulyatsiy.py"
echo "=== s6_6_qadam_cloud_vs_edge_vs_gibrid_kechikis.py ==="; python3 s6_6_qadam_cloud_vs_edge_vs_gibrid_kechikis.py || echo "XATO: s6_6_qadam_cloud_vs_edge_vs_gibrid_kechikis.py"
