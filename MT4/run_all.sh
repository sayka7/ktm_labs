#!/bin/bash
# Barcha bosqichlarni ketma-ket bajaradi
cd "$(dirname "$0")" || exit 1
echo "=== s1_1_qadam_ai_xizmati_fastapi_model_reestri.py ==="; python3 s1_1_qadam_ai_xizmati_fastapi_model_reestri.py || echo "XATO: s1_1_qadam_ai_xizmati_fastapi_model_reestri.py"
echo "=== s2_2_qadam_konfiguratsiya_qatlamlari_chuqur.py ==="; python3 s2_2_qadam_konfiguratsiya_qatlamlari_chuqur.py || echo "XATO: s2_2_qadam_konfiguratsiya_qatlamlari_chuqur.py"
echo "=== s3_3_qadam_dockerfile_va_docker_compose_gen.py ==="; python3 s3_3_qadam_dockerfile_va_docker_compose_gen.py || echo "XATO: s3_3_qadam_dockerfile_va_docker_compose_gen.py"
echo "=== s4_4_qadam_kubernetes_manifestlari_generato.py ==="; python3 s4_4_qadam_kubernetes_manifestlari_generato.py || echo "XATO: s4_4_qadam_kubernetes_manifestlari_generato.py"
echo "=== s5_5_qadam_versiyani_chiqarish_rollingupdat.py ==="; python3 s5_5_qadam_versiyani_chiqarish_rollingupdat.py || echo "XATO: s5_5_qadam_versiyani_chiqarish_rollingupdat.py"
echo "=== s6_6_qadam_hpa_autoscaling_v2_cluster_autos.py ==="; python3 s6_6_qadam_hpa_autoscaling_v2_cluster_autos.py || echo "XATO: s6_6_qadam_hpa_autoscaling_v2_cluster_autos.py"
echo "=== s7_7_qadam_operatsion_hisobot_slo_xato_byud.py ==="; python3 s7_7_qadam_operatsion_hisobot_slo_xato_byud.py || echo "XATO: s7_7_qadam_operatsion_hisobot_slo_xato_byud.py"
