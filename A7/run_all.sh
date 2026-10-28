#!/bin/bash
# Barcha bosqichlarni ketma-ket bajaradi
cd "$(dirname "$0")" || exit 1
echo "=== s1_1_qadam_malumot_sxemasi_validatsiyasi_va.py ==="; python3 s1_1_qadam_malumot_sxemasi_validatsiyasi_va.py || echo "XATO: s1_1_qadam_malumot_sxemasi_validatsiyasi_va.py"
echo "=== s2_2_qadam_model_oqitish_eksperiment_jurnal.py ==="; python3 s2_2_qadam_model_oqitish_eksperiment_jurnal.py || echo "XATO: s2_2_qadam_model_oqitish_eksperiment_jurnal.py"
echo "=== s3_3_qadam_pytorch_mlp_onnx_eksport_va_onnx.py ==="; python3 s3_3_qadam_pytorch_mlp_onnx_eksport_va_onnx.py || echo "XATO: s3_3_qadam_pytorch_mlp_onnx_eksport_va_onnx.py"
echo "=== s4_4_qadam_fastapi_xizmati_predict_healthz_.py ==="; python3 s4_4_qadam_fastapi_xizmati_predict_healthz_.py || echo "XATO: s4_4_qadam_fastapi_xizmati_predict_healthz_.py"
echo "=== s5_5_qadam_monitoring_psi_ks_bilan_drift_an.py ==="; python3 s5_5_qadam_monitoring_psi_ks_bilan_drift_an.py || echo "XATO: s5_5_qadam_monitoring_psi_ks_bilan_drift_an.py"
echo "=== s6_6_qadam_adolatlilik_fairlearn_metricfram.py ==="; python3 s6_6_qadam_adolatlilik_fairlearn_metricfram.py || echo "XATO: s6_6_qadam_adolatlilik_fairlearn_metricfram.py"
echo "=== s7_7_qadam_loyiha_pasporti_va_model_kartasi.py ==="; python3 s7_7_qadam_loyiha_pasporti_va_model_kartasi.py || echo "XATO: s7_7_qadam_loyiha_pasporti_va_model_kartasi.py"
