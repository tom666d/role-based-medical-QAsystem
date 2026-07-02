import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from vector_store import ingest_transcripts


def setup_demo(use_mock: bool = True):
    if use_mock:
        from mock_transcripts import MOCK_DATA
        with open('../transcripts.json', 'w') as f:
            json.dump(MOCK_DATA, f, indent=2)
        print("✓ Mock transcripts written to transcripts.json")
    else:
        from transcription_tool import transcribe_all
        print(transcribe_all())

    result = ingest_transcripts('../transcripts.json')
    print(f"✓ {result}")
    print("\n🚀 Demo ready! Launch AutoGen Studio in this same terminal.")


if __name__ == "__main__":
    setup_demo(use_mock=True)  # Change to False to use real audio + Whisper