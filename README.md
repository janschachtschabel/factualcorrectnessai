# KI-gestütztes Bewertungstool für Sachrichtigkeit von Bildungsinhalten

Eine Python/Streamlit-App zur automatisierten KI-gestützten Qualitätsbewertung von Bildungsinhalten mithilfe von OpenAI-Modellen.

## Hauptfunktionen

- Automatische Bewertung von Bildungsinhalten nach definierten Qualitätskriterien
- Einzelbewertung oder Batch-Verarbeitung von Beschreibungstexten
- Unterstützung verschiedener OpenAI Modelle (gpt-4o-mini, gpt-4o)
- Detaillierte Bewertungsberichte mit Verbesserungsvorschlägen
- Visualisierung der Bewertungsergebnisse durch interaktive Grafiken
- Export der Ergebnisse in CSV oder JSON Format

## Systemvoraussetzungen

- Python 3.8 oder höher
- Internetverbindung für OpenAI API
- OpenAI API-Schlüssel

## Installation

1. Abhängigkeiten installieren:
```bash
pip install -r requirements-app.txt
```

2. Umgebungsvariablen konfigurieren:
   - Erstellen Sie eine `.env` Datei oder setzen Sie die Umgebungsvariablen direkt:
   ```
   OPENAI_API_KEY=ihr_api_schlüssel
   BASE_URL=alternative_api_url  # Optional
   MODEL_ID=gpt-4o-mini         # Optional, Standard ist gpt-4o-mini
   ```

## Verwendung

1. Anwendung starten:
```bash
streamlit run app.py
```

2. Öffnen Sie die angezeigte URL im Browser (standardmäßig http://localhost:8501)

3. Konfigurieren Sie die Einstellungen in der Seitenleiste:
   - Wählen Sie das gewünschte LLM-Modell
   - Geben Sie Ihren API-Schlüssel ein (falls nicht in Umgebungsvariablen gesetzt)
   - Aktivieren Sie bei Bedarf den Entwickler-Modus

4. Laden Sie Ihre JSON-Datei(en) hoch und starten Sie die Bewertung

## Datenformat

### Eingabe-JSON
Die Anwendung erwartet JSON-Dateien mit folgender Struktur:
```json
{
  "node": {
    "properties": {
      "cclom:general_description": ["Ihr Beschreibungstext hier"],
      "sys:node-uuid": ["eindeutige-uuid"]
    }
  }
}
```

### Ausgabeformate
- CSV: Tabellarische Darstellung der Bewertungen mit Scores
- JSON: Detaillierte Ergebnisse inklusive Verbesserungsvorschläge

## Bewertungskriterien

Die Bewertungskriterien werden aus der `config.yaml` geladen und umfassen standardmäßig:

1. Sachliche Richtigkeit
2. Klarheit und Verständlichkeit
3. Objektivität
4. Relevanz
5. Struktur und Länge
6. Sprache und Stil

## Entwickler-Modus

Der Entwickler-Modus zeigt zusätzliche Informationen:
- JSON-Parsing Details
- Extrahierte Texte und UUIDs
- API-Kommunikation
- Verarbeitungsfortschritt

## Fehlerbehebung

1. API-Timeouts
   - Standard-Timeout: 30 Sekunden
   - Lösung: Internetverbindung prüfen
   - Bei großen Batches: Kleinere Datensätze verwenden

2. JSON-Parsing-Fehler
   - Aktivieren Sie den Entwickler-Modus
   - Überprüfen Sie die JSON-Struktur
   - Stellen Sie sicher, dass die Datei UTF-8 kodiert ist

3. Keine Beschreibungen gefunden
   - Überprüfen Sie die JSON-Struktur
   - Stellen Sie sicher, dass die Felder korrekt benannt sind

## Benötigte Dateien

Die Anwendung benötigt folgende Dateien:
- `app.py`: Die Hauptanwendung
- `requirements-app.txt`: Liste der Python-Abhängigkeiten
- `config.yaml`: Konfigurationsdatei für Bewertungskriterien
- `.env`: Optionale Datei für Umgebungsvariablen
