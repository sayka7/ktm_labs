#!/bin/bash
# Barcha bosqichlarni ketma-ket bajaradi
cd "$(dirname "$0")" || exit 1
echo "=== s1_1_qadam_tokenizatsiya_bpe_ni_qolda_oqiti.py ==="; python3 s1_1_qadam_tokenizatsiya_bpe_ni_qolda_oqiti.py || echo "XATO: s1_1_qadam_tokenizatsiya_bpe_ni_qolda_oqiti.py"
echo "=== s2_2_qadam_kv_kesh_xotira_kalkulyatori_va_k.py ==="; python3 s2_2_qadam_kv_kesh_xotira_kalkulyatori_va_k.py || echo "XATO: s2_2_qadam_kv_kesh_xotira_kalkulyatori_va_k.py"
echo "=== s3_3_qadam_bolaklash_chunking_va_qidiruv_bm.py ==="; python3 s3_3_qadam_bolaklash_chunking_va_qidiruv_bm.py || echo "XATO: s3_3_qadam_bolaklash_chunking_va_qidiruv_bm.py"
echo "=== s4_4_qadam_mini_rag_qidiruv_prompt_yigish_e.py ==="; python3 s4_4_qadam_mini_rag_qidiruv_prompt_yigish_e.py || echo "XATO: s4_4_qadam_mini_rag_qidiruv_prompt_yigish_e.py"
echo "=== s5_5_qadam_hallucination_detektori_javob_ju.py ==="; python3 s5_5_qadam_hallucination_detektori_javob_ju.py || echo "XATO: s5_5_qadam_hallucination_detektori_javob_ju.py"
echo "=== s6_6_qadam_prompt_injection_detektori_qoida.py ==="; python3 s6_6_qadam_prompt_injection_detektori_qoida.py || echo "XATO: s6_6_qadam_prompt_injection_detektori_qoida.py"
echo "=== s7_7_qadam_prompt_shablonlari_va_baholovchi.py ==="; python3 s7_7_qadam_prompt_shablonlari_va_baholovchi.py || echo "XATO: s7_7_qadam_prompt_shablonlari_va_baholovchi.py"
