# JAI Version: 0.11.0
"""Persistent memory and structured knowledge learned through communication."""

from __future__ import annotations
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

class MemoryManager:
    CATEGORIES = ("conversations","knowledge","people","concepts","experiences","tasks","skills","procedures","errors","relationships")

    def __init__(self, root: str | Path = "memory") -> None:
        self.root=Path(root); self.raw_root=self.root/"raw"; self.organized_root=self.root/"organized"; self.index_path=self.root/"index.json"
        self.root.mkdir(parents=True,exist_ok=True); self.raw_root.mkdir(parents=True,exist_ok=True); self.organized_root.mkdir(parents=True,exist_ok=True)
        for category in self.CATEGORIES: (self.organized_root/category).mkdir(parents=True,exist_ok=True)
        if not self.index_path.exists(): self._write_json(self.index_path,{"version":"0.11.0","encounters":0,"records":0})

    @staticmethod
    def _timestamp(): return datetime.now(timezone.utc).isoformat()
    @staticmethod
    def _safe_name(value): return re.sub(r"[^a-zA-Z0-9_-]+","_",value.strip().lower()).strip("_")[:80] or "memory"
    @staticmethod
    def _write_json(path,data): path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding="utf-8")
    def _read_index(self):
        try: return json.loads(self.index_path.read_text(encoding="utf-8"))
        except (FileNotFoundError,json.JSONDecodeError): return {"version":"0.11.0","encounters":0,"records":0}
    def _update_index(self,encounter_delta=0,record_delta=0):
        index=self._read_index(); index["version"]="0.11.0"; index["encounters"]=index.get("encounters",0)+encounter_delta; index["records"]=index.get("records",0)+record_delta; self._write_json(self.index_path,index)

    def _category_for(self,text):
        lowered=text.lower()
        rules={
            "errors":("error","failed","failure","bug","broken","exception","traceback"),
            "tasks":("task","todo","need to","have to","must ","should "),
            "procedures":("how to","steps","procedure","first ","then ","click ","open "),
            "skills":("skill","can do","learned how","ability"),
            "relationships":("friend","family","brother","sister","mother","father","relationship","works with","belongs to"),
            "people":("i am","my name","my ","user","person","people"),
            "experiences":("i tried","i did","worked","didn't work","succeeded","success","failed","experience"),
            "concepts":("what is","means","meaning","concept","definition"),
        }
        for category,words in rules.items():
            if any(word in lowered for word in words): return category
        return "knowledge"

    def _keywords(self,text):
        words=re.findall(r"[A-Za-z0-9']+",text.lower()); stop={"the","and","that","this","with","from","have","has","what","when","where","which","would","could","should","about","there","their","they","them","your","you","jai","user","into","then","than","just","like"}
        result=[]; seen=set()
        for word in words:
            if len(word)>=3 and word not in stop and word not in seen: seen.add(word); result.append(word)
        return result[:20]

    def _make_record(self,category,encounter_id,text,response="",metadata=None):
        return {"version":"0.11.0","memory_id":encounter_id,"category":category,"created_at":self._timestamp(),"content":text.strip()[:500],"response":response.strip(),"keywords":self._keywords(text+" "+response),"confidence":0.9 if metadata and metadata.get("type")=="learned_fact" else 0.5,"learned":bool(metadata and metadata.get("type")=="learned_fact"),"learning":metadata or {},"times_recalled":0,"metadata":metadata or {}}

    def remember(self,content,*,response="",source="encounter",metadata=None):
        if not content.strip(): raise ValueError("Memory content cannot be empty.")
        now=datetime.now(timezone.utc); encounter_id=now.strftime("%Y%m%dT%H%M%S%fZ"); category=(metadata or {}).get("category") or self._category_for(content+" "+response)
        record=self._make_record(category,encounter_id,content,response,{"source":source,**(metadata or {})})
        raw={"version":"0.11.0","memory_id":encounter_id,"created_at":record["created_at"],"source":source,"content":content,"response":response,"organized_category":category,"metadata":metadata or {}}
        self._write_json(self.raw_root/now.strftime("%Y")/now.strftime("%m")/f"{encounter_id}.json",raw)
        name=self._safe_name(record["keywords"][0] if record["keywords"] else category)
        self._write_json(self.organized_root/category/f"{encounter_id}_{name}.json",record); self._update_index(1,1); return record

    def remember_conversation(self,user_message,jai_response):
        return self.remember(user_message,response=jai_response,source="conversation",metadata={"type":"conversation_turn"})

    def learn_fact(self,subject,relation,value,*,source="conversation",replace=False):
        subject=subject.strip(); relation=relation.strip(); value=value.strip()
        if not subject or not relation or not value: raise ValueError("Learned facts require subject, relation, and value.")
        if replace:
            for old in self.learned_facts(subject):
                old_id=old.get("memory_id","")
                for path in self.organized_root.rglob(f"{old_id}_*.json"):
                    try: path.unlink()
                    except OSError: pass
                for path in self.raw_root.rglob(f"{old_id}.json"):
                    try: path.unlink()
                    except OSError: pass
        return self.remember(f"{subject} {relation} {value}",source=source,metadata={"type":"learned_fact","category":"knowledge","subject":subject,"relation":relation,"value":value})

    def learn_statement(self,text,*,source="conversation",replace=False):
        """Persist explicit training text when no structured fact can be extracted."""
        if replace:
            normalized=text.strip().lower()
            for category in self.CATEGORIES:
                for path in (self.organized_root/category).rglob("*.json"):
                    try: record=json.loads(path.read_text(encoding="utf-8"))
                    except (OSError,json.JSONDecodeError): continue
                    meta=record.get("metadata",{})
                    if meta.get("type")=="learned_statement" and str(meta.get("statement","")).strip().lower()==normalized:
                        try: path.unlink()
                        except OSError: pass
        return self.remember(text,source=source,metadata={"type":"learned_statement","category":"knowledge","statement":text})

    def learned_facts(self,subject):
        target=subject.strip().lower(); facts=[]
        if not target: return facts
        for category in self.CATEGORIES:
            for path in (self.organized_root/category).rglob("*.json"):
                try: record=json.loads(path.read_text(encoding="utf-8"))
                except (OSError,json.JSONDecodeError): continue
                meta=record.get("metadata",{})
                if meta.get("type")=="learned_fact" and str(meta.get("subject","")).strip().lower()==target: facts.append(record)
        return sorted(facts,key=lambda x:x.get("created_at",""),reverse=True)

    def search(self,query,limit=10):
        if not query.strip(): return []
        query_words=set(self._keywords(query)) or set(re.findall(r"[A-Za-z0-9']+",query.lower()))
        scored=[]
        for path in self.organized_root.rglob("*.json"):
            try: record=json.loads(path.read_text(encoding="utf-8"))
            except (OSError,json.JSONDecodeError): continue
            hay=set(record.get("keywords",[])); hay.update(self._keywords(record.get("content",""))); score=len(query_words&hay)
            if score: scored.append((score,record.get("created_at",""),record))
        scored.sort(key=lambda x:(x[0],x[1]),reverse=True); results=[x[2] for x in scored[:max(0,limit)]]
        for record in results: self._increment_recall(record.get("memory_id",""))
        return results

    def _increment_recall(self,memory_id):
        if not memory_id: return
        for path in self.organized_root.rglob(f"{memory_id}_*.json"):
            try: record=json.loads(path.read_text(encoding="utf-8"))
            except (OSError,json.JSONDecodeError): return
            record["times_recalled"]=record.get("times_recalled",0)+1; self._write_json(path,record); return

    def recall_context(self,query,limit=5):
        memories=self.search(query,limit)
        if not memories: return ""
        lines=["Relevant memories:"]
        for memory in memories:
            content=memory.get("content","").replace("\n"," "); response=memory.get("response","").replace("\n"," ")
            lines.append(f"- {content} -> JAI previously replied: {response}" if response else f"- {content}")
        return "\n".join(lines)

    def stats(self):
        index=self._read_index()
        return {"version":"0.11.0","root":str(self.root),"encounters":index.get("encounters",0),"records":index.get("records",0),"categories":{c:len(list((self.organized_root/c).glob("*.json"))) for c in self.CATEGORIES}}
