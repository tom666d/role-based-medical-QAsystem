import json

MOCK_DATA = [
  {
    "patient_id": "P001",
    "summary": "Patient presented with persistent cough and chest tightness for two weeks.",
    "diagnosis": "Mild bronchitis.",
    "prescription": "Amoxicillin 500mg twice daily for 7 days. Rest and fluids."
  },
  {
    "patient_id": "P002",
    "summary": "Patient has elevated LDL cholesterol at 180, above recommended level.",
    "diagnosis": "Hypercholesterolemia. Dietary changes required.",
    "prescription": "Atorvastatin 20mg once daily at bedtime."
  },
  {
    "patient_id": "P003",
    "summary": "Annual checkup. Blood pressure elevated at 145/90. Stress-related.",
    "diagnosis": "Mild hypertension, likely stress-related.",
    "prescription": "Lisinopril 10mg once daily. Reduce sodium intake."
  }
]

with open('../transcripts.json', 'w') as f:
    json.dump(MOCK_DATA, f, indent=2)
print("Mock transcripts saved to transcripts.json")