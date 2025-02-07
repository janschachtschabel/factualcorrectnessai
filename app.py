import streamlit as st
import json
import os
import yaml
from openai import AsyncOpenAI
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import requests
from typing import List, Dict, Any

# Konfiguration
MODEL_ID = os.getenv("MODEL_ID", "gpt-4o-mini")
API_KEY = os.environ.get("OPENAI_API_KEY")  # Hole den Key aus den Systemumgebungsvariablen
BASE_URL = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")  # Optional: Base URL

# OpenAI Client initialisieren
client = AsyncOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    timeout=30.0  # Erhöhe Timeout für längere Verarbeitungszeiten
)

def load_criteria():
    """Lade Bewertungskriterien aus config.yaml"""
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config.get('evaluation_criteria', [])
    except Exception as e:
        st.error(f"Fehler beim Laden der Kriterien: {str(e)}")
        return []

# Bewertungskriterien aus config.yaml laden
CRITERIA = load_criteria()

EVALUATION_PROMPT = """
Du bist ein KI-gestützter Richter / Evaluator für Beschreibungstexte einer Online-Lernplattform. Analysiere die Texte und bewerte sie kritisch basierend auf den vorgegebenen Metriken. Basierend auf dem Kontext und den Bewertungskriterien, generiere einen neuen Beschreibungstext. Falls bereits eine Beschreibung vorhanden ist, verbessere diese unter Berücksichtigung der Kriterien.

Bewertungskriterien:
{criteria_text}

Zu bewertender Text:
{text}

Bewertungsschritte:
1. Lies den bereitgestellten Beschreibungstext sorgfältig durch.
2. Gehe jedes Bewertungskriterium einzeln durch und bewerte, ob der Text die Kriterien erfüllt.
3. Begründe deine Bewertung für jedes Kriterium.
4. Erstelle eine Reflektion über die Stärken und Schwächen des Textes.
5. Generiere einen verbesserten Beschreibungstext, der:
   - Sachlich und informell ist
   - Keine Superlative enthält
   - Nicht emotional oder werblich ist
   - Für eine seriöse Bildungsplattform geeignet ist
   - Auf dramaturgische Satzkonstrukte wie "Erleben Sie..." oder "Entdecken Sie..." verzichtet

Ausgabeformat:
{criteria_format}
"""

def debug_print(message):
    """Debug-Ausgabe wenn Debug-Modus aktiv"""
    if st.session_state.get('debug_mode', False):
        st.write(f"DEBUG: {message}")

async def evaluate_text(text, model_id, api_key):
    """Bewerte einen Text mit dem OpenAI API"""
    try:
        if not text or text.strip() == "":
            st.error("Leerer Text kann nicht bewertet werden!")
            return None
            
        criteria_text, criteria_format = format_criteria_for_prompt()
        prompt = EVALUATION_PROMPT.format(
            criteria_text=criteria_text,
            text=text,
            criteria_format=criteria_format
        )
        
        # Erstelle einen neuen Client mit dem aktuellen API Key
        client = AsyncOpenAI(
            base_url=os.getenv("BASE_URL"),
            api_key=api_key,
            timeout=30.0  # Erhöhe Timeout auf 30 Sekunden
        )
        
        response = await client.chat.completions.create(
            model=model_id,  # Verwende das ausgewählte Modell
            messages=[
                {"role": "system", "content": "Du bist ein präziser Bewerter von Bildungsinhalten."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        st.error(f"Fehler bei der Bewertung: {str(e)}")
        return None

def load_json_file(uploaded_file):
    """Lade JSON-Datei"""
    try:
        content = uploaded_file.read()
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        data = json.loads(content)
        
        # Wenn es ein einzelnes Objekt ist, mache eine Liste daraus
        if isinstance(data, dict):
            data = [data]
            
        debug_print(f"JSON geladen: {len(data)} Einträge gefunden")
        debug_print(f"Erster Eintrag hat Node: {'node' in data[0]}")
        return data
    except Exception as e:
        st.error(f"Fehler beim Laden der JSON-Datei: {str(e)}")
        return None

def extract_description(json_data):
    """Extrahiere die Beschreibung aus den JSON-Daten"""
    try:
        if isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data
            
        # Prüfe ob wir ein "node" Objekt haben
        if 'node' in data:
            data = data['node']
            
        if 'properties' in data:
            properties = data['properties']
            if 'cclom:general_description' in properties and properties['cclom:general_description']:
                description = properties['cclom:general_description'][0]
                debug_print(f"Beschreibung gefunden: {description[:100]}...")
                return description
            else:
                debug_print("Keine Beschreibung im Feld 'cclom:general_description' gefunden")
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        debug_print(f"Fehler bei der Beschreibungsextraktion: {str(e)}")
    return None

def extract_node_uuid(json_data):
    """Extrahiere die Node-UUID aus den JSON-Daten"""
    try:
        if isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data
            
        # Prüfe ob wir ein "node" Objekt haben
        if 'node' in data:
            data = data['node']
            
        if 'properties' in data:
            properties = data['properties']
            if 'sys:node-uuid' in properties and properties['sys:node-uuid']:
                uuid = properties['sys:node-uuid'][0]
                debug_print(f"UUID gefunden: {uuid}")
                return uuid
            else:
                debug_print("Keine UUID im Feld 'sys:node-uuid' gefunden")
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        debug_print(f"Fehler bei der UUID-Extraktion: {str(e)}")
    return None

def format_criteria_for_prompt():
    """Formatiere die Kriterien für den Prompt"""
    criteria_text = ""
    criteria_format = {
        "scores": {},
        "reflection": "Reflektion über die Stärken und Schwächen des Textes",
        "improved_text": "Verbesserter Beschreibungstext"
    }
    
    for criterion in CRITERIA:
        criteria_text += f"- {criterion['name']}: {criterion['description']}\n"
        criteria_format["scores"][criterion['name']] = {
            "score": "0 oder 1",
            "reasoning": "Begründung für die Bewertung"
        }
    
    return criteria_text, json.dumps(criteria_format, indent=2, ensure_ascii=False)

def create_metrics_plots(results_df):
    """Erstelle verschiedene Visualisierungen für die Ergebnisse"""
    
    # Durchschnittliche Scores pro Kriterium
    fig_avg = px.bar(
        results_df.mean().reset_index(),
        x='index',
        y=0,
        title='Durchschnittliche Scores pro Kriterium',
        labels={'index': 'Kriterium', '0': 'Score'}
    )
    st.plotly_chart(fig_avg)
    
    # Boxplot für Score-Verteilung
    fig_box = px.box(
        results_df.melt(),
        y='value',
        x='variable',
        title='Score-Verteilung pro Kriterium',
        labels={'variable': 'Kriterium', 'value': 'Score'}
    )
    st.plotly_chart(fig_box)
    
    # Korrelationsmatrix
    corr_matrix = results_df.corr()
    fig_corr = px.imshow(
        corr_matrix,
        title='Korrelationsmatrix der Kriterien',
        labels=dict(color="Korrelation")
    )
    st.plotly_chart(fig_corr)

def save_results(results, original_data):
    """Speichere die Bewertungsergebnisse als JSON"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "metadata": {
            "timestamp": timestamp,
            "model": MODEL_ID,
            "criteria": [c['name'] for c in CRITERIA]
        },
        "evaluations": []
    }
    
    for i, (result, original) in enumerate(zip(results, original_data)):
        node_id = original.get("node", {}).get("ref", {}).get("id", f"unknown_{i}")
        evaluation = {
            "node_id": node_id,
            "timestamp": timestamp,
            "scores": result["scores"],
            "reflection": result.get("reflection", ""),
            "improved_text": result.get("improved_text", "")
        }
        output["evaluations"].append(evaluation)
    
    # Speichere die Ergebnisse
    filename = f"evaluations_{timestamp}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    return filename

async def process_batch(texts, progress_bar, original_data, model_id, api_key):
    """Verarbeite einen Batch von Texten"""
    results = []
    
    for i, text in enumerate(texts):
        try:
            result = await evaluate_text(text, model_id, api_key)
            if result:
                results.append(result)
            else:
                st.warning(f"Keine Ergebnisse für Text {i+1}")
        except Exception as e:
            st.error(f"Fehler bei der Verarbeitung von Text {i+1}: {str(e)}")
            results.append({"error": str(e)})
        finally:
            progress_bar.progress((i + 1) / len(texts))
    
    return results

def convert_score(score):
    """Konvertiert einen Score in einen numerischen Wert"""
    if isinstance(score, (int, float)):
        return float(score)
    try:
        return float(score)
    except (ValueError, TypeError):
        # Wenn der Score nicht konvertiert werden kann, geben wir 0 zurück
        return 0.0

class EduSharingAPI:
    def __init__(self, base_url):
        self.base_url = base_url

    def get_child_collections(self, repository, collection_id, max_items=500, skip_count=0, property_filter=None):
        url = f"{self.base_url}/collection/v1/collections/{repository}/{collection_id}/children/collections"
        params = {
            "scope": "MY",
            "fetchCounts": "true",
            "maxItems": max_items,
            "skipCount": skip_count,
            "propertyFilter": property_filter or "-all-"
        }
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            return response.json().get('collections', [])
        else:
            st.error(f"Fehler beim Abrufen der Kind-Sammlungen für Collection ID {collection_id}: {response.text}")
            return []

    def get_all_collections(self, repository, collection_ids, visited_collections=None, max_results=None, property_filter=None):
        if visited_collections is None:
            visited_collections = set()
        collections = []

        collection_queue = collection_ids.copy()
        while collection_queue:
            collection_id = collection_queue.pop(0)
            if collection_id in visited_collections:
                continue
            visited_collections.add(collection_id)
            child_collections = self.get_child_collections(repository, collection_id, property_filter=property_filter)
            collections.append(collection_id)
            child_collection_ids = [col['ref']['id'] for col in child_collections]
            collection_queue.extend(child_collection_ids)
            if max_results and len(collections) >= max_results:
                break
        return collections

    def get_collection_contents(self, repository, collection_id, max_items=500, skip_count=0, property_filter=None):
        url = f"{self.base_url}/node/v1/nodes/{repository}/{collection_id}/children"
        params = {
            "maxItems": max_items,
            "skipCount": skip_count,
            "filter": "files",
            "propertyFilter": property_filter or "-all-"
        }
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            return response.json().get('nodes', [])
        else:
            st.error(f"Fehler beim Abrufen der Inhalte für Collection ID {collection_id}: {response.text}")
            return []

def fetch_wlo_collection_data(api, repository, root_collection_ids, max_results=None, skip_count=0, all_content=False, property_filter=None):
    st.info(f"Sammle alle Sammlungen für Collection IDs: {', '.join(root_collection_ids)}")
    all_collection_ids = api.get_all_collections(repository, root_collection_ids, max_results=None, property_filter=property_filter)

    # Sicherstellen, dass die initialen Collection IDs enthalten sind
    all_collection_ids = list(set(root_collection_ids + all_collection_ids))

    st.info(f"{len(all_collection_ids)} Sammlungen gefunden")

    # Alle gesammelten Inhalte sammeln
    all_contents = []
    total_contents_fetched = 0

    for idx, collection_id in enumerate(all_collection_ids):
        if not all_content and max_results and total_contents_fetched >= max_results:
            break
        st.info(f"Abrufen der Inhalte für Sammlung {idx + 1} von {len(all_collection_ids)}: {collection_id}")
        contents = []
        collection_skip_count = skip_count
        while True:
            if not all_content and max_results and total_contents_fetched >= max_results:
                break
            fetch_limit = 100  # Anzahl der Inhalte pro Anfrage
            remaining = max_results - total_contents_fetched if max_results else fetch_limit
            max_items = min(fetch_limit, remaining)
            fetched_contents = api.get_collection_contents(repository, collection_id, max_items=max_items, skip_count=collection_skip_count, property_filter=property_filter)
            if not fetched_contents:
                break
            contents.extend(fetched_contents)
            total_contents_fetched += len(fetched_contents)
            collection_skip_count += len(fetched_contents)
            if len(fetched_contents) < fetch_limit:
                break
        all_contents.extend(contents)
    return all_contents

def save_results_to_json(results, filename, identifier=None):
    # Erstelle output Verzeichnis falls es nicht existiert
    import os
    os.makedirs('output', exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if identifier:
        identifier = identifier.replace(" ", "_").replace("/", "_")
        filename_with_timestamp = f"{filename.rstrip('.json')}_{identifier}_{timestamp}.json"
    else:
        filename_with_timestamp = f"{filename.rstrip('.json')}_{timestamp}.json"
    filepath = os.path.join('output', filename_with_timestamp)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    st.success(f"Ergebnisse wurden in output/{filename_with_timestamp} gespeichert")
    return filename_with_timestamp

def get_fields(data: Any, parent_key: str = '') -> List[str]:
    fields = []
    if isinstance(data, dict):
        for key, value in data.items():
            full_key = f"{parent_key}.{key}" if parent_key else key
            fields.append(full_key)
            fields.extend(get_fields(value, full_key))
    elif isinstance(data, list) and len(data) > 0:
        for item in data:
            fields.extend(get_fields(item, parent_key))
    return list(set(fields))

def main():
    st.title("Bewertungstool für Sachrichtigkeit")
    
    # Sidebar für Konfiguration
    with st.sidebar:
        st.header("Konfiguration")
        
        # Modellauswahl
        model_id = st.selectbox(
            "KI Modell auswählen",
            options=["gpt-4o-mini", "gpt-4o"],
            index=0,
            help="Wählen Sie das zu verwendende KI-Modell aus"
        )
        
        # API Key Eingabe
        default_api_key = os.getenv("OPENAI_API_KEY", "")
        api_key = st.text_input(
            "OpenAI API Key",
            value=default_api_key,
            type="password",
            help="Der API Key wird aus der Umgebungsvariable OPENAI_API_KEY geladen, falls gesetzt. Alternativ können Sie hier einen Key eingeben."
        )
        
        # Verwende den eingegebenen API Key, falls vorhanden, sonst den aus den Umgebungsvariablen
        if api_key:
            API_KEY = api_key
        elif default_api_key:
            st.info("API Key aus Umgebungsvariable OPENAI_API_KEY geladen")
            API_KEY = default_api_key
        else:
            API_KEY = None
        
        st.markdown("---")
        
        # Debug-Modus
        debug_mode = st.checkbox("Debug-Modus", value=False)
        st.session_state['debug_mode'] = debug_mode

    if not API_KEY:
        st.error("Kein API-Key gefunden! Bitte geben Sie einen API-Key ein oder setzen Sie die Umgebungsvariable OPENAI_API_KEY.")
        return

    # Kriterien anzeigen
    if CRITERIA:
        with st.expander("Bewertungskriterien"):
            for criterion in CRITERIA:
                st.markdown(f"**{criterion['name']}**")
                st.write(criterion['description'])
    else:
        st.error("Keine Bewertungskriterien geladen! Bitte stelle sicher, dass config.yaml vorhanden ist.")
        return

    # Hauptbereich in Tabs aufteilen
    tab1, tab2, tab3 = st.tabs(["Einzeltext", "Batch-Bewertung einer WLO-JSON", "WLO Sammlung"])
    
    # Tab 1: Einzeltext-Bewertung
    with tab1:
        st.subheader("Einzeltext bewerten")
        input_text = st.text_area("Text zur Bewertung eingeben", height=200, key="single_text_input")
        single_button = st.button("Text bewerten", key="single_text_button", type="primary")
        
        if single_button and input_text:
            debug_print("Starte Einzeltext-Bewertung")
            try:
                with st.spinner("Bewertung läuft..."):
                    result = asyncio.run(evaluate_text(input_text, model_id, API_KEY))
                    if result:
                        st.success("Bewertung abgeschlossen!")
                        
                        # Ergebnisse anzeigen
                        st.subheader("Bewertungsergebnisse")
                        
                        # Scores
                        st.write("### Scores")
                        for criterion, details in result["scores"].items():
                            score = details["score"]
                            reasoning = details["reasoning"]
                            st.write(f"**{criterion}**: {score}")
                            st.write(f"*Begründung*: {reasoning}")
                            st.write("---")
                        
                        # Reflektion
                        st.write("### Reflektion")
                        st.write(result["reflection"])
                        
                        # Verbesserter Text
                        st.write("### Verbesserungsvorschlag")
                        st.write(result["improved_text"])
                        
                        # Download Buttons
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                "Als JSON speichern",
                                data=json.dumps(result, indent=2, ensure_ascii=False),
                                file_name="einzelbewertung.json",
                                mime="application/json",
                                key="single_json_download"
                            )
                        with col2:
                            # CSV Export
                            csv_data = {
                                "Kriterium": [],
                                "Score": [],
                                "Begründung": []
                            }
                            for criterion, details in result["scores"].items():
                                csv_data["Kriterium"].append(criterion)
                                csv_data["Score"].append(details["score"])
                                csv_data["Begründung"].append(details["reasoning"])
                                
                            df = pd.DataFrame(csv_data)
                            st.download_button(
                                "Als CSV speichern",
                                data=df.to_csv(index=False),
                                file_name="einzelbewertung.csv",
                                mime="text/csv",
                                key="single_csv_download"
                            )
                        
            except Exception as e:
                st.error(f"Fehler bei der Bewertung: {str(e)}")
        elif single_button:
            st.warning("Bitte geben Sie einen Text zur Bewertung ein.")

    # Tab 2: Batch-Bewertung
    with tab2:
        st.subheader("Batch-Bewertung")
        uploaded_file = st.file_uploader("JSON-Datei hochladen", type=['json'], key="batch_file_upload")
        if uploaded_file:
            batch_size = st.number_input("Anzahl der Nodes", min_value=1, max_value=100, value=10, key="batch_size_input")
            process_all = st.checkbox("Alle Nodes verarbeiten (ignoriert Anzahl)", value=False, key="process_all_nodes")
            
            batch_button = st.button("Batch-Bewertung starten", key="batch_button", type="primary")
            
            if batch_button:
                debug_print("Datei wurde hochgeladen")
                data = load_json_file(uploaded_file)
                if data:
                    debug_print("Batch-Bewertung gestartet")
                    # Wenn nicht alle Nodes verarbeitet werden sollen, beschränke auf batch_size
                    if not process_all:
                        data = data[:batch_size]
                        debug_print(f"Verarbeite {len(data)} von {len(data)} Nodes")
                    else:
                        debug_print(f"Verarbeite alle {len(data)} Nodes")
                    
                    texts = []
                    node_uuids = []
                    
                    debug_print(f"Verarbeite {len(data)} Items")
                    for item in data:
                        debug_print(f"Verarbeite Item: {item.get('node', {}).get('title', 'Kein Titel')}")
                        
                        description = extract_description(item)
                        node_uuid = extract_node_uuid(item)
                        
                        if description and node_uuid:
                            texts.append(description)
                            node_uuids.append(node_uuid)
                            debug_print(f"Text und UUID extrahiert: {node_uuid}")
                            debug_print(f"Beschreibung: {description[:100]}...")
                        else:
                            debug_print("Konnte Text oder UUID nicht extrahieren")
                            if not description:
                                debug_print("Keine Beschreibung gefunden")
                            if not node_uuid:
                                debug_print("Keine UUID gefunden")
                    
                    debug_print(f"Extrahierte Texte: {len(texts)}")
                    if texts:
                        debug_print(f"Starte Batch-Verarbeitung mit {len(texts)} Texten")
                        st.subheader(f"Gefundene Beschreibungen: {len(texts)}")
                        
                        try:
                            progress_bar = st.progress(0)
                            with st.spinner("Bewertung läuft..."):
                                results = asyncio.run(process_batch(texts, progress_bar, data, model_id, API_KEY))
                                debug_print("Batch-Verarbeitung abgeschlossen")
                        except Exception as e:
                            st.error(f"Fehler bei der Batch-Verarbeitung: {str(e)}")
                            return

                        if results:
                            st.subheader("Bewertungsergebnisse:")
                            
                            # Rohdaten der Bewertungen
                            with st.expander("Rohdaten der Bewertungen"):
                                st.json(results)
                            
                            # Scores und Begründungen
                            all_scores = []
                            all_reasonings = []
                            all_reflections = []
                            all_improvements = []
                            
                            for i, (result, node_uuid) in enumerate(zip(results, node_uuids)):
                                scores = {k: convert_score(v["score"]) for k, v in result["scores"].items()}
                                scores["node_uuid"] = node_uuid
                                all_scores.append(scores)
                                
                                reasonings = {k: v["reasoning"] for k, v in result["scores"].items()}
                                reasonings["node_uuid"] = node_uuid
                                all_reasonings.append(reasonings)
                                
                                all_reflections.append({
                                    "node_uuid": node_uuid,
                                    "reflection": result.get("reflection", ""),
                                    "original_text": texts[i]
                                })
                                
                                all_improvements.append({
                                    "node_uuid": node_uuid,
                                    "improved_text": result.get("improved_text", ""),
                                    "original_text": texts[i]
                                })
                            
                            scores_df = pd.DataFrame(all_scores).set_index("node_uuid")
                            reasonings_df = pd.DataFrame(all_reasonings).set_index("node_uuid")
                            
                            # Metriken anzeigen
                            metrics_cols = st.columns(3)
                            for i, criterion in enumerate(CRITERIA):
                                with metrics_cols[i % 3]:
                                    criterion_name = criterion['name']
                                    if criterion_name in scores_df.columns:
                                        st.metric(
                                            criterion_name,
                                            f"{scores_df[criterion_name].mean():.2f}"
                                        )
                            
                            # Visualisierungen
                            create_metrics_plots(scores_df)
                            
                            # Detaillierte Ergebnisse
                            st.subheader("Scores pro Text")
                            st.dataframe(scores_df)
                            
                            st.subheader("Begründungen")
                            st.dataframe(reasonings_df)
                            
                            # Reflektionen und Verbesserungen
                            st.subheader("Reflektionen und Verbesserungen")
                            for reflection in all_reflections:
                                with st.expander(f"Node: {reflection['node_uuid']}"):
                                    st.markdown("**Reflektion:**")
                                    st.write(reflection['reflection'])
                                    st.markdown("**Verbesserter Text:**")
                                    st.write(all_improvements[all_reflections.index(reflection)]['improved_text'])
                            
                            # Ergebnisse speichern
                            filename = save_results(results, data)
                            st.success(f"Ergebnisse wurden in {filename} gespeichert!")
                            
                            # Export-Optionen
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            
                            # Konvertiere DataFrames zu CSV-Strings
                            scores_csv = scores_df.to_csv().encode('utf-8')
                            reasonings_csv = reasonings_df.to_csv().encode('utf-8')
                            reflections_df = pd.DataFrame(all_reflections)
                            improvements_df = pd.DataFrame(all_improvements)
                            reflections_csv = reflections_df.to_csv().encode('utf-8')
                            improvements_csv = improvements_df.to_csv().encode('utf-8')
                            
                            # Download-Buttons in Spalten anordnen
                            col1, col2 = st.columns(2)
                            with col1:
                                st.download_button(
                                    label="Scores als CSV herunterladen",
                                    data=scores_csv,
                                    file_name=f"scores_{timestamp}.csv",
                                    mime="text/csv"
                                )
                                st.download_button(
                                    label="Reflektionen als CSV herunterladen",
                                    data=reflections_csv,
                                    file_name=f"reflections_{timestamp}.csv",
                                    mime="text/csv"
                                )
                            with col2:
                                st.download_button(
                                    label="Begründungen als CSV herunterladen",
                                    data=reasonings_csv,
                                    file_name=f"reasonings_{timestamp}.csv",
                                    mime="text/csv"
                                )
                                st.download_button(
                                    label="Verbesserungen als CSV herunterladen",
                                    data=improvements_csv,
                                    file_name=f"improvements_{timestamp}.csv",
                                    mime="text/csv"
                                )
                    
    # Tab 3: WLO Sammlung
    with tab3:
        st.subheader("WLO Sammlung")
        
        # WLO API Konfiguration
        base_url = "https://redaktion.openeduhub.net/edu-sharing/rest"
        repository = "-home-"
        
        # Eingabefeld für die Sammlungs-ID
        collection_id = st.text_input("WLO Sammlungs-ID")
        
        # Optionen für die Verarbeitung
        batch_size = st.number_input("Anzahl der Nodes", min_value=1, max_value=100, value=10, key="batch_size_input_wlo")
        process_all = st.checkbox("Alle Nodes verarbeiten (ignoriert Anzahl)", value=False, key="process_all_nodes_wlo")
        include_subcollections = st.checkbox("Untersammlungen rekursiv einbeziehen", value=False, key="include_subcollections_wlo")
        
        if st.button("Sammlung abrufen und bewerten", key="fetch_collection_button", type="primary"):
            if not collection_id:
                st.error("Bitte geben Sie eine Sammlungs-ID ein.")
            else:
                api = EduSharingAPI(base_url)
                max_results = None if process_all else int(batch_size)
                
                try:
                    st.info(f"Rufe Sammlung {collection_id} ab...")
                    if include_subcollections:
                        st.info("Sammle alle Untersammlungen...")
                        collections = api.get_all_collections(repository, [collection_id])
                        st.info(f"{len(collections)} Sammlungen gefunden")
                        contents = []
                        for idx, coll_id in enumerate(collections):
                            st.info(f"Verarbeite Sammlung {idx + 1} von {len(collections)}: {coll_id}")
                            coll_contents = fetch_wlo_collection_data(
                                api,
                                repository,
                                [coll_id],
                                max_results=max_results,
                                skip_count=0,
                                all_content=process_all,
                                property_filter="-all-"
                            )
                            contents.extend(coll_contents)
                    else:
                        contents = fetch_wlo_collection_data(
                            api,
                            repository,
                            [collection_id],
                            max_results=max_results,
                            skip_count=0,
                            all_content=process_all,
                            property_filter="-all-"
                        )
                    
                    if contents:
                        st.info(f"{len(contents)} Nodes gefunden")
                        
                        # Extrahiere die Texte und UUIDs aus den Nodes
                        texts = []
                        node_uuids = []
                        for node in contents:
                            if "properties" in node:
                                # Extrahiere den Text aus den Properties
                                description = node["properties"].get("cclom:general_description", [""])[0]
                                if description:
                                    texts.append(description)
                                    node_uuids.append(node.get("ref", {}).get("id", ""))
                        
                        if texts:
                            st.info(f"Starte Batch-Bewertung für {len(texts)} Texte...")
                            
                            # Initialisiere Progress Bar
                            progress_bar = st.progress(0)
                            
                            # Führe die Batch-Bewertung durch
                            results = []
                            for idx, (text, node_uuid) in enumerate(zip(texts, node_uuids)):
                                progress = (idx + 1) / len(texts)
                                progress_bar.progress(progress)
                                st.info(f"Bewerte Text {idx + 1} von {len(texts)}")
                                
                                # Führe die Bewertung durch
                                try:
                                    evaluation = asyncio.run(evaluate_text(text, model_id, API_KEY))
                                    # Konvertiere alle Scores zu float
                                    for criterion in evaluation["scores"]:
                                        evaluation["scores"][criterion]["score"] = float(evaluation["scores"][criterion]["score"])
                                    evaluation["node_uuid"] = node_uuid
                                    results.append(evaluation)
                                except Exception as e:
                                    st.error(f"Fehler bei der Bewertung von Text {idx + 1}: {str(e)}")
                            
                            # Entferne Progress Bar nach Abschluss
                            progress_bar.empty()
                            
                            # Speichere die Ergebnisse
                            if results:
                                filename = save_results_to_json(results, 'evaluations.json', identifier=collection_id)
                                
                                # Erstelle und zeige die Zusammenfassung
                                st.subheader("Zusammenfassung der Bewertungen")
                                
                                # Berechne durchschnittliche Scores für jedes Kriterium
                                avg_scores = {}
                                for criterion in CRITERIA:
                                    criterion_name = criterion['name']
                                    scores = [result["scores"][criterion_name]["score"] for result in results]
                                    avg_scores[criterion_name] = sum(scores) / len(scores)
                                
                                # Zeige die Metriken in Spalten
                                metrics_cols = st.columns(3)
                                for i, criterion in enumerate(CRITERIA):
                                    with metrics_cols[i % 3]:
                                        criterion_name = criterion['name']
                                        st.metric(
                                            criterion_name,
                                            f"{avg_scores[criterion_name]:.2f}"
                                        )
                                
                                # Biete Download-Optionen an
                                st.subheader("Ergebnisse herunterladen")
                                
                                # Erstelle DataFrames für die verschiedenen CSV-Exporte
                                scores_data = []
                                reasonings_data = []
                                reflections_data = []
                                improvements_data = []
                                
                                for result in results:
                                    node_uuid = result["node_uuid"]
                                    
                                    # Scores und Begründungen
                                    for criterion, details in result["scores"].items():
                                        scores_data.append({
                                            "Node UUID": node_uuid,
                                            "Kriterium": criterion,
                                            "Score": details["score"]
                                        })
                                        reasonings_data.append({
                                            "Node UUID": node_uuid,
                                            "Kriterium": criterion,
                                            "Begründung": details["reasoning"]
                                        })
                                    
                                    # Reflektionen
                                    reflections_data.append({
                                        "Node UUID": node_uuid,
                                        "Reflektion": result["reflection"]
                                    })
                                    
                                    # Verbesserungen
                                    improvements_data.append({
                                        "Node UUID": node_uuid,
                                        "Verbesserter Text": result["improved_text"]
                                    })
                                
                                # Erstelle DataFrames
                                scores_df = pd.DataFrame(scores_data)
                                reasonings_df = pd.DataFrame(reasonings_data)
                                reflections_df = pd.DataFrame(reflections_data)
                                improvements_df = pd.DataFrame(improvements_data)
                                
                                # Download Buttons
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.download_button(
                                        "Scores als CSV herunterladen",
                                        scores_df.to_csv(index=False),
                                        f"scores_{collection_id}.csv",
                                        mime="text/csv"
                                    )
                                    st.download_button(
                                        "Reflektionen als CSV herunterladen",
                                        reflections_df.to_csv(index=False),
                                        f"reflections_{collection_id}.csv",
                                        mime="text/csv"
                                    )
                                with col2:
                                    st.download_button(
                                        "Begründungen als CSV herunterladen",
                                        reasonings_df.to_csv(index=False),
                                        f"reasonings_{collection_id}.csv",
                                        mime="text/csv"
                                    )
                                    st.download_button(
                                        "Verbesserungen als CSV herunterladen",
                                        improvements_df.to_csv(index=False),
                                        f"improvements_{collection_id}.csv",
                                        mime="text/csv"
                                    )
                                
                                # Detaillierte Ergebnisse in ausklappbaren Bereichen
                                st.subheader("Detaillierte Bewertungen")
                                for result in results:
                                    with st.expander(f"Bewertung für Node {result['node_uuid']}", expanded=False):
                                        st.write("### Scores")
                                        for criterion, details in result["scores"].items():
                                            st.write(f"**{criterion}**: {details['score']:.2f}")
                                            st.write(f"*Begründung*: {details['reasoning']}")
                                            st.write("---")
                                        
                                        st.write("### Reflektion")
                                        st.write(result["reflection"])
                                        
                                        st.write("### Verbesserungsvorschlag")
                                        st.write(result["improved_text"])
                        else:
                            st.warning("Keine Beschreibungstexte in den Nodes gefunden.")
                    else:
                        st.warning("Keine Inhalte in der Sammlung gefunden.")
                except Exception as e:
                    st.error(f"Fehler beim Abrufen der Sammlung: {str(e)}")
                    
if __name__ == "__main__":
    main()
