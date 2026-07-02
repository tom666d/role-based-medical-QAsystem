import whisper, json, os, openai
from dotenv import load_dotenv

load_dotenv()

def extract_sections(transcript: str, patient_id: str) -> dict:
    """Use GPT-4.1 to split a raw transcript into summary, diagnosis, prescription."""
    client = openai.AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ["AZURE_API_VERSION"]
    )
    response = client.chat.completions.create(
        model=os.environ["AZURE_DEPLOYMENT_NAME"],
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract three fields from this doctor-patient transcript as JSON. "
                    "Return ONLY valid JSON with keys: summary, diagnosis, prescription. "
                    "If a field is not present, return an empty string for that key."
                )
            },
            {"role": "user", "content": transcript}
        ],
        response_format={"type": "json_object"}
    )
    data = json.loads(response.choices[0].message.content)
    data["patient_id"] = patient_id
    return data


def transcribe_all(audio_dir: str = 'audio_files') -> str:
    """Transcribe all audio files and extract structured sections."""
    model = whisper.load_model("base")
    records = []

    for fname in sorted(os.listdir(audio_dir)):
        if not fname.endswith(('.mp3', '.wav')):
            continue
        patient_id = fname.split('_')[0]   # e.g. P001_visit.mp3 → P001
        fpath = os.path.join(audio_dir, fname)

        print(f"Transcribing {fname}...")
        result = model.transcribe(fpath)
        structured = extract_sections(result["text"], patient_id)
        records.append(structured)

    # Load existing records and only add new patients — never overwrite
    existing = []
    if os.path.exists("transcripts.json"):
        with open("transcripts.json") as f:
            existing = json.load(f)

    existing_ids = {r["patient_id"] for r in existing}
    new_records = [r for r in records if r["patient_id"] not in existing_ids]

    with open("transcripts.json", "w") as f:
        json.dump(existing + new_records, f, indent=2)

    return f"Transcribed {len(new_records)} new file(s). transcripts.json now has {len(existing + new_records)} patient(s)."


if __name__ == "__main__":
    print(transcribe_all())