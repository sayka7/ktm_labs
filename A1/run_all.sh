#!/bin/bash
# Barcha bosqichlarni ketma-ket bajaradi
cd "$(dirname "$0")" || exit 1
echo "=== s1_1_qadam_talablardan_yuklama_hisobi_gflop.py ==="; python3 s1_1_qadam_talablardan_yuklama_hisobi_gflop.py || echo "XATO: s1_1_qadam_talablardan_yuklama_hisobi_gflop.py"
echo "=== s2_2_qadam_platformalar_jadvali_va_talablar.py ==="; python3 s2_2_qadam_platformalar_jadvali_va_talablar.py || echo "XATO: s2_2_qadam_platformalar_jadvali_va_talablar.py"
echo "=== s3_3_qadam_kechikish_byudjeti_va_monte_karl.py ==="; python3 s3_3_qadam_kechikish_byudjeti_va_monte_karl.py || echo "XATO: s3_3_qadam_kechikish_byudjeti_va_monte_karl.py"
echo "=== s4_4_qadam_arifmetik_intensivlik_va_rooflin.py ==="; python3 s4_4_qadam_arifmetik_intensivlik_va_rooflin.py || echo "XATO: s4_4_qadam_arifmetik_intensivlik_va_rooflin.py"
echo "=== s5_5_qadam_kichik_cnn_uchun_flop_va_paramet.py ==="; python3 s5_5_qadam_kichik_cnn_uchun_flop_va_paramet.py || echo "XATO: s5_5_qadam_kichik_cnn_uchun_flop_va_paramet.py"
echo "=== s6_6_qadam_hardware_software_co_design_izla.py ==="; python3 s6_6_qadam_hardware_software_co_design_izla.py || echo "XATO: s6_6_qadam_hardware_software_co_design_izla.py"
echo "=== s7_7_qadam_edge_va_markazlashgan_arxitektur.py ==="; python3 s7_7_qadam_edge_va_markazlashgan_arxitektur.py || echo "XATO: s7_7_qadam_edge_va_markazlashgan_arxitektur.py"
