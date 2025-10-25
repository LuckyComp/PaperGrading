import fitz
import numpy as np
import re
from sentence_transformers import SentenceTransformer, util
import json

class FeatureExtractor:
    def __init__(self, document_name:str, sen_model:SentenceTransformer|None = None):
        self.document = fitz.open(document_name)
        self.section_wise_content = []
        self.__extract_text()
        self.sen_model = sen_model if sen_model else SentenceTransformer("allenai/scibert_scivocab_uncased")

    def __extract_text(self) -> None:
        page_text = ""
        try:
            with fitz.open(self.document) as doc:
                for page in doc:
                    page_text += page.get_text()
        except RuntimeError as e:
            if "too many kids in page tree" in str(e):
                doc = fitz.open(self.document)
                doc.save(self.document, garbage=4, clean=True)
                doc.close()            
                print(f"Successfully repaired and saved to {self.document}")
                self.__extract_text()
        self._extract_content(page_text)
    
    def _extract_content(self, text: str) -> None:        
        keyword_map = {
            "abstract": ["abstract"],
            "introduction": ["introduction"],
            "methodology": [
                "method", "methods", "methodology", "approach",
                "experimental setup", "experimental design", "experiment", "experiments",
                "related work", "analysis", "materials and methods"
            ],
            "results": ["result", "results", "evaluation", "experimental results"],
            "conclusion": ["conclusion", "conclusions", "discussion", "future work"],
            "references": ["reference", "references", "bibliography", "citations"]
        }

        lines = text.split('\n')
        section_boundaries = []


        for i, line in enumerate(lines):
            stripped_line = line.strip()
            if not stripped_line:
                continue

            line_lower = stripped_line.lower()

            if len(stripped_line.split()) > 15: # Relaxed limit
                continue

            found_title_key = None
            matched_keyword = None # Keep track of which keyword matched

            for title_key, keywords in keyword_map.items():
                for keyword in keywords:
                    pattern = r"^\s*(\d+\.?\s*)?" + re.escape(keyword) + r"\s*($|\W)"

                    match = re.search(pattern, line_lower)

                    if match and match.start() == 0:
                         found_title_key = title_key
                         matched_keyword = keyword # Store the keyword that matched
                         break # Stop checking keywords for this title_key
                if found_title_key:
                    break # Stop checking title_keys for this line


            if found_title_key:
                if found_title_key not in [key for _, key in section_boundaries]:
                    section_boundaries.append((i, found_title_key))

        self.section_wise_content = []
        if not section_boundaries:
            return

        section_boundaries.append((len(lines), "end_of_document"))

        for i in range(len(section_boundaries) - 1):
            start_index, section_key = section_boundaries[i]
            end_index, _ = section_boundaries[i+1]
            content_lines = lines[start_index + 1 : end_index]
            section_content = "\n".join(
                [line.strip() for line in content_lines if line.strip()]
            )
            self.section_wise_content.append({
                "section_title": section_key,
                "section_content": section_content
            })
            
    def get_encodings(self) -> np.array:
        encodings = [0,0,0,0]
        for section in self.section_wise_content:
            if section["section_title"] == "abstract":
                encodings[0] = self.sen_model.encode(section["section_content"])
            elif section["section_title"] == "introduction":
                encodings[1] = self.sen_model.encode(section["section_content"])
            elif section["section_title"] == "methodology":
                encodings[2] = self.sen_model.encode(section["section_content"])
            elif section["section_title"] == "conclusion" or section["section_title"] == "results":
                encodings[3] = self.sen_model.encode(section["section_content"])
        if len(encodings) == 4:
            return encodings
        else:
            raise Exception("Couldn't make feature encoding. Too few sections.")
        
    def get_references_count(self) -> int:
        reference_section = None
        for section in self.section_wise_content:
            if section["section_title"] == "references":
                reference_section = section
        if not reference_section:
            return None
        references = re.split(r"\n(?=[A-Z][a-z]+,)", reference_section["section_content"])
        return len(references)

    def __get_similarity_scores(self, phrase_one:str, phrase_two:str) -> float:
        sentences = [phrase_one, phrase_two]
        embeddings = self.sen_model.encode(sentences, convert_to_tensor=True)
        cosine_sim_scores = util.cos_sim(embeddings[0], embeddings[1])
        return float(cosine_sim_scores[0])
        
    def get_coherence_score(self):
        intro_section_content, conclusion_section_content = (None, None)
        for section in self.section_wise_content:
            if "introduction" in section["section_title"]:
                intro_section_content = section["section_content"]
            elif "conclusion" in section["section_title"] or "discussion" in section["section_title"]:
                conclusion_section_content = section["section_content"]
        if intro_section_content and conclusion_section_content:
            return self.__get_similarity_scores(intro_section_content, conclusion_section_content)
        else: return 0.0

    def get_abstract_quality(self):
        abs_section, methodology_section = (None, None)
        for section in self.section_wise_content:
            if "abstract" == section["section_title"]:
                abs_section = section["section_content"]
            elif "methodology" == section["section_title"]:
                methodology_section = section["section_content"]
        if abs_section and methodology_section:
            return self.__get_similarity_scores(methodology_section, abs_section)
        else: return 0.0

if __name__ == "__main__":
    fe = FeatureExtractor(document_name=r"", output_file="")
    print(fe.get_coherence_score())
    print(fe.get_compressed_encodings())
    print(fe.get_references_count())
    print(fe.get_abstract_quality())