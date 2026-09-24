"""
AI智能伴侣 - Agent 版
基于本文件夹各章节的 LangChain 技术改造：
- chapter01: init_chat_model 初始化 DeepSeek 模型
- chapter05/07: @tool 定义工具，create_agent 创建智能体（工具调用需关闭思考模式）
- chapter07-06: agent.stream(stream_mode="messages") 流式输出
- chapter08: 长期记忆（remember/recall 工具 + JSON 持久化）
- chapter09/知识库项目: Milvus + RAG（pymilvus 建库/切分/嵌入/upsert/COSINE 检索）；
  知识库 knowledge.txt 为【现成开源数据集】加工而成（非手写）：
  HuggingFace sunorme/chinese-adorable-high-emotional-intelligence-chat（170 条高情商恋爱对话），
  原始 JSON 存 raw_data/，加工脚本见 build_knowledge.py，手写版备份为 knowledge_handwritten.txt；
  嵌入源降级链 SiliconFlow Qwen3-Embedding-8B(4096维) → DashScope(1024维) → 关键词检索；
  LangSmith 监控（.env 中 LANGSMITH_* 配置，运行记录见项目 1_langchain）

运行方式:
    streamlit run app.py
"""
import os
import json
import re
import time
import uuid
import hashlib
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timedelta

import streamlit as st
import streamlit.components.v1 as components

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, PIIMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import AIMessageChunk, ToolMessage, SystemMessage
from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, Field

# ============================== 全局配置 ==============================
# app 在子目录运行，.env 在项目根目录：find_dotenv 从本文件向上逐级查找
load_dotenv(find_dotenv(usecwd=True), override=True)

# ============================== 模型初始化（chapter01 + 多模型容灾） ==============================
# 支持多个模型依次尝试——DeepSeek 余额耗尽自动切 DashScope，SiliconFlow 兜底
# @st.cache_resource 保证只在进程启动时探活一次，后续所有 rerun 直接复用（不会再调 LLM）
MODELS = [
    {"name": "deepseek-v4-flash",  "prefix": "deepseek",   "key": "DEEPSEEK_API_KEY",   "url": "DEEPSEEK_BASE_URL",   "extra": {"thinking": {"type": "disabled"}}},
    {"name": "qwen-plus",         "prefix": "dashscope",  "key": "DASHSCOPE_API_KEY",  "url": None,                  "extra": None},
    {"name": "qwen-turbo",        "prefix": "dashscope",  "key": "DASHSCOPE_API_KEY",  "url": None,                  "extra": None},
    {"name": "Qwen/Qwen3-8B",     "prefix": "openai",     "key": "SILICONFLOW_API_KEY","url": "SILICONFLOW_BASE_URL","extra": None},
]

@st.cache_resource(show_spinner=False)
def _init_model():
    for _m in MODELS:
        _key = os.getenv(_m["key"])
        if not _key: continue
        try:
            _kwargs = {"model": f"{_m['prefix']}:{_m['name']}", "api_key": _key}
            if _m["url"] and os.getenv(_m["url"]): _kwargs["base_url"] = os.getenv(_m["url"])
            if _m["extra"]: _kwargs["extra_body"] = _m["extra"]
            _mdl = init_chat_model(**_kwargs)
            _mdl.invoke("ping")  # 轻量探活
            return _mdl, f"{_m['prefix']}:{_m['name']}"
        except Exception as _e:
            print(f"[{_m['prefix']}:{_m['name']}] 不可用：{_e.__class__.__name__}")
    return None, None

model, ACTIVE_MODEL_NAME = _init_model()
if model is None:
    st.error("所有 LLM 都不可用：请检查 DEEPSEEK_API_KEY / DASHSCOPE_API_KEY / SILICONFLOW_API_KEY 余额")
    st.stop()

# ============================== 全局路径 ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SESSION_DIR = os.path.join(BASE_DIR, "sessions")
MEMORY_FILE = os.path.join(BASE_DIR, "long_memory.json")   # 长期记忆持久化文件（跨会话共享）
PENDING_MEMORY_FILE = os.path.join(BASE_DIR, "pending_memories.json")  # 待确认记忆持久化（防刷新丢失）
KNOWLEDGE_FILE = os.path.join(BASE_DIR, "knowledge.txt")   # 恋爱知识库（RAG 数据源）

# 性格模板：一键切换人设，选"自定义"后可在侧边栏自由编辑
PERSONA_PRESETS = {
    "温柔体贴": "温柔似水、善解人意，说话轻声细语，总能在细节里关心人，偶尔撒娇",
    "活泼开朗": "元气满满、爱开玩笑，聊天像小太阳，擅长活跃气氛、接梗玩梗",
    "高冷傲娇": "嘴硬心软、说话带点小毒舌，不轻易夸人但其实很在意对方",
    "知性大方": "成熟稳重、见识广，像知心大姐姐一样给人建议和安全感",
    "自定义": "",
}

# ============================== 页面配置 ==============================
st.set_page_config(
    page_title="AI智能伴侣",
    page_icon="💞",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={},
)


# ============================== 可靠 toast（fragment 连续局部刷新时 st.toast 会被前端去重，
# 改用 JS 直接向主文档 body 注入气泡：每次调用都新建节点+重播动画，连续点击每次必弹） =========
def custom_toast(message: str, icon: str = "✨", duration_ms: int = 2600):
    # json.dumps 安全转义后注入 JS，避免消息里的引号/换行破坏脚本
    js_msg = json.dumps(str(message), ensure_ascii=False)
    js_icon = json.dumps(icon, ensure_ascii=False)
    components.html(
        f"""
        <script>
        (function(){{
          var pdoc = window.parent.document;
          var box = pdoc.createElement('div');
          box.style.cssText = 'position:fixed;right:24px;bottom:24px;z-index:999999;'
            + 'background:rgba(38,39,48,.96);color:#fff;padding:12px 18px;border-radius:12px;'
            + 'box-shadow:0 8px 28px rgba(0,0,0,.35);max-width:340px;display:flex;gap:8px;'
            + 'align-items:center;pointer-events:none;'
            + "font:14px/1.5 -apple-system,'Segoe UI','Microsoft YaHei',sans-serif;";
          var ic = pdoc.createElement('span');
          ic.textContent = {js_icon};
          ic.style.cssText = 'font-size:18px;';
          var tx = pdoc.createElement('span');
          tx.textContent = {js_msg};
          box.appendChild(ic); box.appendChild(tx);
          pdoc.body.appendChild(box);
          box.animate([
            {{opacity:0, transform:'translateY(12px) scale(.95)'}},
            {{opacity:1, transform:'none', offset:0.12}},
            {{opacity:1, offset:0.88}},
            {{opacity:0, transform:'translateY(8px)'}}
          ], {{duration:{duration_ms}, easing:'ease', fill:'forwards'}});
          setTimeout(function(){{ box.remove(); }}, {duration_ms + 200});
        }})();
        </script>
        """,
        height=0,
    )


# ============================== 常驻 JS 指令通道（toast / 新建会话即时清场） ==============================
# 关键兼容点：当前 Streamlit 版本里 components.html 的 iframe 作为"全新 delta 节点"首次挂载时，
# srcdoc 里的 <script> 不会执行；只有 iframe 已存在、srcdoc 被更新时脚本才可靠运行。
# 因此 fragment 末尾【始终】渲染同一个通道 iframe（页面加载即挂载，空指令），
# 之后所有指令都走"更新 srcdoc"路径下发；已执行指令 id 记在父窗口上做幂等去重。
_JS_CHANNEL_HEAD = """
<script>
(function(){
  var win = window.parent;
  if (!win.__jsCmdDone) win.__jsCmdDone = {};
  var cmds ="""

_JS_CHANNEL_TAIL = """
  function showToast(c){
    var pdoc = win.document;
    var box = pdoc.createElement('div');
    box.style.cssText = 'position:fixed;right:24px;bottom:24px;z-index:999999;'
      + 'background:rgba(38,39,48,.96);color:#fff;padding:12px 18px;border-radius:12px;'
      + 'box-shadow:0 8px 28px rgba(0,0,0,.35);max-width:340px;display:flex;gap:8px;'
      + 'align-items:center;pointer-events:none;'
      + "font:14px/1.5 -apple-system,'Segoe UI','Microsoft YaHei',sans-serif;";
    var ic = pdoc.createElement('span');
    ic.textContent = c.icon;
    ic.style.cssText = 'font-size:18px;';
    var tx = pdoc.createElement('span');
    tx.textContent = c.msg;
    box.appendChild(ic); box.appendChild(tx);
    pdoc.body.appendChild(box);
    box.animate([
      {opacity:0, transform:'translateY(12px) scale(.95)'},
      {opacity:1, transform:'none', offset:0.12},
      {opacity:1, offset:0.88},
      {opacity:0, transform:'translateY(8px)'}
    ], {duration:c.dur, easing:'ease', fill:'forwards'});
    setTimeout(function(){ box.remove(); }, c.dur + 200);
  }
  // 新建会话后只做 fragment 级刷新（避免 st.rerun() 整页灰屏+空白等待）：
  // JS 即时隐藏聊天气泡、输入框上方插欢迎条、同步顶部标题；只切 body class/加外来节点，
  // 绝不删除 React 管理的 DOM，聊天 fragment 下次重跑时自行清场。
  function clearChat(c){
    var doc = win.document;
    if (!doc.getElementById('__new_chat_style__')) {
      var st = doc.createElement('style');
      st.id = '__new_chat_style__';
      st.textContent =
        "body.__chat_cleared__ [data-testid='stMain'] [data-testid='stChatMessage']"
        + "{display:none !important;}"
        + "#__new_chat_welcome__{display:flex;gap:8px;align-items:center;margin:0 0 14px;"
        + "padding:14px 18px;border:1px dashed rgba(255,255,255,.22);border-radius:14px;"
        + "color:rgba(255,255,255,.72);font:15px/1.6 -apple-system,'Segoe UI','Microsoft YaHei',sans-serif;}";
      doc.head.appendChild(st);
    }
    doc.body.classList.add('__chat_cleared__');
    var old = doc.getElementById('__new_chat_welcome__');
    if (old) old.remove();
    var input = doc.querySelector("[data-testid='stMain'] [data-testid='stChatInput']");
    if (input) {
      var w = doc.createElement('div');
      w.id = '__new_chat_welcome__';
      w.textContent = '✨ 已开启新会话，发条消息打个招呼吧～';
      input.parentElement.insertBefore(w, input);
    }
    // 顶部 caption（🎯 标题 · 会话文件）在主区域 fragment 之外，整页不重跑时手动同步
    var cap = doc.querySelector("[data-testid='stMain'] [data-testid='stCaptionContainer']");
    if (cap) cap.textContent = '🎯 ' + c.title + '  ·  会话文件: ' + c.name;
  }
  // 清场时机由服务端掌握：chat_display fragment 在"清场后第一条消息"那次 run 里下发 cleanup。
  // 注意：该指令在 run 中途就到达，而 React 要等 fragment run 结束才卸载旧气泡 DOM——
  // 若立刻移除 class，旧会话气泡会闪现。这里先记录当前气泡节点，轮询到它们全部被 React
  // 卸载（新内容已提交）后再解除隐藏；waitForSwap=false 用于整页 rerun 后的立即恢复。
  function cleanupChat(waitForSwap){
    var doc = win.document;
    function finish(){
      doc.body.classList.remove('__chat_cleared__');
      var w0 = doc.getElementById('__new_chat_welcome__');
      if (w0) w0.remove();
    }
    var oldNodes = Array.prototype.slice.call(
      doc.querySelectorAll("[data-testid='stMain'] [data-testid='stChatMessage']"));
    if (waitForSwap === false || !oldNodes.length) { finish(); return; }
    var started = Date.now();
    var timer = setInterval(function(){
      var swapped = oldNodes.every(function(n){ return !n.isConnected; });
      if (swapped || Date.now() - started > 8000) { clearInterval(timer); finish(); }
    }, 60);
  }
  // 切换/加载已有会话：遮罩 class 和欢迎条是 JS 挂在 body 上的外来状态，
  // st.rerun() 的 React 重渲染不会清它们，必须显式移除，否则旧会话消息被 CSS 隐藏成空白
  function restoreChat(c){
    cleanupChat(false);
    var doc = win.document;
    // caption 曾被 clearChat 用 JS 直接改过文本，React 认为"没变"不会改回，这里手动同步
    var caps = doc.querySelectorAll("[data-testid='stMain'] [data-testid='stCaptionContainer']");
    if (caps.length) caps[caps.length-1].textContent = '🎯 ' + c.title + '  ·  会话文件: ' + c.name;
  }
  for (var i=0;i<cmds.length;i++){
    var c = cmds[i];
    if (!c || win.__jsCmdDone[c.id]) continue;
    win.__jsCmdDone[c.id] = 1;
    try {
      if (c.kind === 'toast') showToast(c);
      else if (c.kind === 'clear') clearChat(c);
      else if (c.kind === 'cleanup') cleanupChat();
      else if (c.kind === 'restore') restoreChat(c);
    } catch(e) {}
  }
})();
</script>
"""


def _js_chan_append(cmd: dict):
    """向常驻 JS 通道追加一条指令（toast/clear），带单调递增 id 供前端幂等去重"""
    st.session_state['_js_chan_seq'] = st.session_state.get('_js_chan_seq', 0) + 1
    cmd['id'] = st.session_state['_js_chan_seq']
    st.session_state.setdefault('_js_chan_pending', []).append(cmd)


def queue_toast(message: str, icon: str = "✨", duration_ms: int = 2600):
    """按钮回调里先入队再 rerun：rerun 后由 fragment 末尾的常驻 JS 通道统一弹出"""
    _js_chan_append({'kind': 'toast', 'msg': str(message), 'icon': icon, 'dur': duration_ms})


def queue_clear_chat(session_name: str, title: str = "主对话"):
    """新建会话即时清场指令（隐藏气泡+欢迎条+同步标题）；
    同时挂起 cleanup：等 chat_display 在清场后第一条消息的那次 run 里下发，恢复气泡显示"""
    _js_chan_append({'kind': 'clear', 'name': str(session_name), 'title': str(title)})
    st.session_state['_chat_cleanup_pending'] = True


def queue_restore_chat(session_name: str, title: str = "主对话"):
    """切换/加载已有会话时下发：移除新建会话遮罩与欢迎条、同步 caption，保证旧会话正常可见"""
    _js_chan_append({'kind': 'restore', 'name': str(session_name), 'title': str(title)})


def render_js_channel(cmds):
    """渲染常驻 JS 通道 iframe【每次 fragment run 都要调用，空指令也调】：
    iframe 常驻存活，指令随 srcdoc 更新可靠下发（首挂不执行脚本是本版 Streamlit 的已知行为）"""
    components.html(
        _JS_CHANNEL_HEAD + json.dumps(cmds or [], ensure_ascii=False) + _JS_CHANNEL_TAIL,
        height=0,
    )


# 整页首次加载时兜底清理历史残留遮罩（处理旧版本脚本留下的 body class）。
# 用 parent window 标志保证本页面生命周期内只跑一次：iframe 被重新挂载时不得重复执行，
# 否则会把常驻通道 clearChat 刚加上的遮罩立刻清掉。
def js_reset_chat_guard():
    components.html(
        """
        <script>
        (function(){
          var win = window.parent;
          if (win.__chatGuardReady) return;
          win.__chatGuardReady = true;
          var doc = win.document;
          doc.body.classList.remove('__chat_cleared__');
          var w = doc.getElementById('__new_chat_welcome__');
          if (w) w.remove();
          if (win.__newChatObserver__) {
            try { win.__newChatObserver__.disconnect(); } catch(e) {}
            win.__newChatObserver__ = null;
          }
        })();
        </script>
        """,
        height=0,
    )

# ============================== 文档加载与切分（chapter09） ==============================
def _bigrams(s: str) -> set:
    s = "".join(ch for ch in s if ch.isalnum())
    return {s[i:i + 2] for i in range(len(s) - 1)} or {s}


@st.cache_resource
def load_knowledge() -> list:
    """加载恋爱知识库：TextLoader + RecursiveCharacterTextSplitter（chapter09）。"""
    if not os.path.exists(KNOWLEDGE_FILE):
        return []
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    docs = TextLoader(file_path=KNOWLEDGE_FILE, encoding="utf-8").load()
    # 现成数据集（HF 高情商对话）每条最长约 150 字，chunk_size 给到 200 保证一条完整不被硬切
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    return [c.page_content for c in splitter.split_documents(docs)]


def keyword_search(query: str, k: int = 3) -> list:
    """关键词兜底检索（Milvus 不可用时使用）"""
    chunks = load_knowledge()
    if not chunks:
        return []
    q_grams = _bigrams(query)
    scored = sorted(((len(q_grams & _bigrams(t)), i) for i, t in enumerate(chunks)), reverse=True)
    hits = [i for score, i in scored[:k] if score > 0]
    return [chunks[i] for i in hits] if hits else chunks[:k]


# ============================== Milvus 向量库（知识库项目/1.py 已跑通的架构） ==============================
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_DB = "companion_db"  # 伴侣专用数据库
EMBED_BATCH = 10            # 嵌入接口每批条数

# 嵌入源降级链：优先用章节里跑通的硅基流动 Qwen3-Embedding-8B(4096维)；
# 不可用时降级 DashScope text-embedding-v3(1024维)。维度不同，各自独立集合。
EMBED_PROVIDERS = [
    {
        "name": "SiliconFlow",
        "model": "Qwen/Qwen3-Embedding-8B",
        "dim": 4096,
        "key_env": "SILICONFLOW_API_KEY",
        "base_url": os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
        "collection": "love_knowledge_qwen3",
    },
    {
        "name": "DashScope",
        "model": "text-embedding-v3",
        "dim": 1024,
        "key_env": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "collection": "love_knowledge",
    },
]


class MilvusLoveRAG:
    """Milvus + RAG 管理器（pymilvus 直连，写法与知识库项目/1.py 一致）"""

    def __init__(self, provider: dict):
        from pymilvus import MilvusClient
        from openai import OpenAI
        self.provider = provider
        self.collection = provider["collection"]
        self.embed_model = provider["model"]
        self.dim = provider["dim"]
        # 1. 连接 Milvus，建库（不存在才建）并切库
        self.client = MilvusClient(uri=MILVUS_URI)
        if MILVUS_DB not in self.client.list_databases():
            self.client.create_database(db_name=MILVUS_DB)
        self.client.use_database(db_name=MILVUS_DB)
        # 2. 建集合（维度与嵌入模型严格一致，COSINE 相似度）
        if not self.client.has_collection(collection_name=self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                dimension=self.dim,
                metric_type="COSINE",
            )
        # 3. 嵌入模型（OpenAI 兼容接口）
        self.embedder = OpenAI(
            api_key=os.getenv(provider["key_env"]),
            base_url=provider["base_url"],
        )

    def embed(self, texts: list) -> list:
        """文本 → 向量（分批调用），并做维度校验"""
        vectors = []
        for i in range(0, len(texts), EMBED_BATCH):
            resp = self.embedder.embeddings.create(
                model=self.embed_model,
                input=texts[i:i + EMBED_BATCH],
            )
            vectors.extend(item.embedding for item in resp.data)
        if any(len(v) != self.dim for v in vectors):
            raise ValueError(f"嵌入维度与集合维度 {self.dim} 不一致")
        return vectors

    @staticmethod
    def _hash_id(text: str) -> int:
        """由文本内容生成稳定主键，重复同步时 upsert 不产生重复行"""
        import hashlib
        return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:15], 16)

    def ingest_file(self, recreate: bool = False) -> int:
        """加载 knowledge.txt → 切分 → 嵌入 → upsert 进 Milvus

        recreate=True 时先删集合再重建（整体更换知识库时使用，
        与知识库项目/1.py 的"先 drop 再 create"写法一致，避免旧条目残留）
        """
        if recreate and self.client.has_collection(collection_name=self.collection):
            self.client.drop_collection(collection_name=self.collection)
            self.client.create_collection(
                collection_name=self.collection,
                dimension=self.dim,
                metric_type="COSINE",
            )
        chunks = load_knowledge()
        if not chunks:
            return 0
        vectors = self.embed(chunks)
        data = [{
            "id": self._hash_id(chunk),
            "vector": vectors[i],
            "text": chunk,
            "source": os.path.basename(KNOWLEDGE_FILE),
        } for i, chunk in enumerate(chunks)]
        self.client.upsert(collection_name=self.collection, data=data)
        self.client.flush(collection_name=self.collection)
        return len(data)

    def search(self, query: str, k: int = 3) -> list:
        """问题向量化 → Milvus COSINE 检索，返回 [(文本, 相似度分, 来源), ...]"""
        query_vector = self.embed([query])[0]
        results = self.client.search(
            collection_name=self.collection,
            data=[query_vector],
            limit=k,
            output_fields=["text", "source"],
        )
        return [(
            hit["entity"]["text"],
            hit["distance"],
            hit["entity"].get("source", "unknown"),
        ) for hit in results[0]]

    def row_count(self) -> int:
        """真实行数（get_collection_stats 在 upsert 后会因删+插未压缩而虚高，用 count(*) 查询）"""
        try:
            result = self.client.query(
                collection_name=self.collection,
                filter="",
                output_fields=["count(*)"],
            )
            return result[0]["count(*)"]
        except Exception:
            return self.client.get_collection_stats(self.collection).get("row_count", 0)


# Milvus 调用专用单线程池：用 future.result(timeout=) 给不可强杀的 gRPC 调用
# 套一层墙钟超时，Milvus 503/卡顿时连接与行数查询都只留在后台线程，不阻塞界面
_MILVUS_COUNT_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix='milvus-cnt')


@st.cache_resource
def get_milvus_rag():
    """Milvus 单例。按 EMBED_PROVIDERS 顺序尝试可用嵌入源；
    全部失败（Milvus 未启动 / 无任何 key）时返回 None，自动降级关键词检索。
    构造（list_databases/has_collection 等 gRPC）也限时 2.5 秒：超时即判定不可用并缓存
    None，避免侧边栏每次 rerun 都被卡十几秒（503 时单次 gRPC 要 9 秒才报错）。"""
    for provider in EMBED_PROVIDERS:
        if not os.getenv(provider["key_env"]):
            continue
        try:
            return _MILVUS_COUNT_POOL.submit(MilvusLoveRAG, provider).result(timeout=2.5)
        except FuturesTimeoutError:
            return None  # 连不上就别再试了，本进程内直接走关键词检索降级
        except Exception:
            continue
    return None


@st.cache_data(ttl=60, show_spinner=False)
def milvus_row_count(collection_name: str):
    """行数查询走短时缓存；1.5 秒超时。返回 None 表示 Milvus 不可用/超时
    （UI 显示降级提示而非永久"统计中"），TTL 过期后自动重试。"""
    rag = get_milvus_rag()
    if rag is None:
        return 0
    try:
        return int(_MILVUS_COUNT_POOL.submit(rag.row_count).result(timeout=1.5))
    except (Exception, FuturesTimeoutError):
        return None


# ============================== 长期记忆持久化（chapter08 思想） ==============================
def load_memories() -> list:
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_memories(memories: list):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memories, f, ensure_ascii=False, indent=2)


def load_pending_memories() -> list:
    """读取待确认记忆队列（落盘，刷新/重启不丢）"""
    if os.path.exists(PENDING_MEMORY_FILE):
        try:
            with open(PENDING_MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_pending_memories(pending: list):
    with open(PENDING_MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(pending, f, ensure_ascii=False, indent=2)


# ============================== 心情晴雨表（chapter06/07-04 的结构化输出） ==============================
class EmotionReport(BaseModel):
    """心情分析报告的结构化定义"""
    score: int = Field(description="用户当前心情评分，0（非常糟糕）到 100（非常好）")
    mood: str = Field(description="用 2-4 个字概括当前情绪，例如 委屈 / 平静 / 开心")
    summary: str = Field(description="一句话总结用户最近的情绪状态")
    suggestion: str = Field(description="给 AI 伴侣的一条关怀建议，怎么说才能让用户开心起来")


def analyze_emotion(messages: list) -> EmotionReport:
    """把最近对话交给 LLM 做结构化情感分析，返回 EmotionReport。
    只取最近 10 轮、每条截断 200 字：情绪看最近语境即可，减少 token 加快返回。"""
    convo = "\n".join(
        f"{'用户' if m['role'] == 'user' else '伴侣'}：{m['content'][:200]}"
        for m in messages[-10:]
    )
    analyzer = model.with_structured_output(EmotionReport)
    return analyzer.invoke([
        {"role": "system", "content": "你是情感分析师，只依据对话内容客观评估用户当前心情。"},
        {"role": "user", "content": f"以下是用户和 AI 伴侣的最近对话，请评估用户当前心情：\n{convo}"},
    ])


# ============================== 工具定义（chapter05/07 的 @tool 写法） ==============================
@tool
def get_time_info(query_type: str = "current") -> str:
    """获取时间相关信息，聊到时间、日期、星期时使用

    Args:
        query_type: 查询类型，可选 "current"(当前时间) / "date"(今天日期) /
            "tomorrow"(明天) / "yesterday"(昨天) / "weekday"(星期几)
    """
    now = datetime.now()
    if query_type == "current":
        return now.strftime("当前时间：%Y年%m月%d日 %H:%M:%S")
    if query_type == "date":
        return now.strftime("今天是：%Y年%m月%d日")
    if query_type == "tomorrow":
        return (now + timedelta(days=1)).strftime("明天是：%Y年%m月%d日")
    if query_type == "yesterday":
        return (now - timedelta(days=1)).strftime("昨天是：%Y年%m月%d日")
    if query_type == "weekday":
        weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        return f"今天是{weekdays[now.weekday()]}"
    return f"不支持的查询类型：{query_type}"


@tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气，关心用户冷暖、提醒穿衣时使用

    Args:
        city (str): 城市名称，例如 "北京"、"重庆"、"杭州"
    """
    # WMO 天气代码 → 中文描述（open-meteo 标准代码表）
    weather_codes = {
        0: "晴", 1: "基本晴", 2: "多云", 3: "阴",
        45: "雾", 48: "雾凇",
        51: "小毛毛雨", 53: "毛毛雨", 55: "大毛毛雨",
        61: "小雨", 63: "中雨", 65: "大雨",
        71: "小雪", 73: "中雪", 75: "大雪", 77: "雪粒",
        80: "阵雨", 81: "中阵雨", 82: "强阵雨",
        85: "小阵雪", 86: "大阵雪",
        95: "雷阵雨", 96: "雷阵雨伴冰雹", 99: "强雷阵雨伴冰雹",
    }
    # 内置模拟数据：真实天气服务不可用时降级使用（chapter05 原版）
    weather_mock = {
        "北京": "多云，15-22℃，空气质量良，湿度 45%",
        "上海": "晴天，18-25℃，空气质量优，湿度 60%",
        "深圳": "小雨，22-28℃，空气质量优，湿度 75%",
        "成都": "阴天，16-23℃，空气质量良，湿度 70%",
        "杭州": "晴天，17-24℃，空气质量优，湿度 55%",
        "广州": "多云，21-29℃，空气质量良，湿度 72%",
    }
    try:
        import requests
        # 第一步：城市名 → 经纬度（open-meteo 地理编码接口，免费无需 key）
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "zh", "format": "json"},
            timeout=5,
        ).json()
        if not geo.get("results"):
            return f"没有查询到城市 {city}，请确认名称（例如：北京/重庆/杭州）"
        loc = geo["results"][0]
        # 第二步：经纬度 → 实时天气
        w = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": loc["latitude"], "longitude": loc["longitude"],
                "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
                "timezone": "auto",
            },
            timeout=5,
        ).json()["current"]
        desc = weather_codes.get(w["weather_code"], f"天气代码{w['weather_code']}")
        return (f"{loc.get('name', city)}当前天气（实时数据）：{desc}，"
                f"气温 {w['temperature_2m']}℃，湿度 {w['relative_humidity_2m']}%，"
                f"风速 {w['wind_speed_10m']} km/h")
    except Exception as e:
        # 网络/API 异常时降级：6 个演示城市返回模拟数据，其余如实告知
        result = weather_mock.get(city)
        if result:
            return f"{city}: {result}（模拟数据，实时天气服务暂不可用）"
        return f"天气服务暂不可用（{e.__class__.__name__}），暂无 {city} 的天气数据"


@tool
def remember_user_info(fact: str) -> str:
    """把关于用户的重要信息提交到待确认队列（生日、喜好、纪念日、工作、心愿等）。
    用户在侧边栏点确认后才会真正写入长期记忆（Human-in-the-Loop，对应 chapter07-09 思路）

    Args:
        fact (str): 要记住的一条信息，一句话描述，例如 "用户的生日是5月20日"
    """
    fact = fact.strip()
    # 去重：已确认的长期记忆和待确认队列里都不能重复
    if any(m["fact"] == fact for m in load_memories()):
        return f"这条已经记过了：{fact}"
    pending = st.session_state.setdefault("pending_memories", [])
    if any(p["fact"] == fact for p in pending):
        return f"这条正在等待用户确认：{fact}"
    pending.append({
        "id": uuid.uuid4().hex[:8],
        "fact": fact,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
    })
    save_pending_memories(pending)  # 落盘：刷新/重启后待确认卡片不丢
    return f"已提交待确认：{fact}（用户在侧边栏点✅后才会写入长期记忆）"


@tool
def recall_user_info(query: str) -> str:
    """按话题检索关于用户的长期记忆（用户提过的喜好、生日、纪念日等）

    Args:
        query (str): 想回忆的话题，例如 "生日" / "喜欢吃什么"
    """
    memories = load_memories()
    if not memories:
        return "暂时没有关于用户的记忆。"
    # 关键词匹配长期记忆（长期记忆量小，关键词足够）
    q_grams = _bigrams(query)
    scored = sorted((
        (len(q_grams & _bigrams(m["fact"])), i)
        for i, m in enumerate(memories)
    ), reverse=True)
    hits = [i for score, i in scored[:5] if score > 0] or list(range(min(5, len(memories))))
    lines = [f"- {memories[i]['fact']}（记录于 {memories[i]['time']}）" for i in hits]
    return "相关记忆：\n" + "\n".join(lines)


@tool
def search_love_knowledge(query: str) -> str:
    """检索恋爱沟通知识库（安慰技巧、吵架处理、仪式感、哄人方法等）

    当用户表达情绪或情感需求时使用：难过、累、加班、被骂、压力大、抱怨、
    吵架、冷战、生病、吃醋、撒娇、求安慰、求哄、聊纪念日等。

    Args:
        query (str): 检索主题，例如 "怎么安慰加班的用户"
    """
    rag = get_milvus_rag()
    if rag is not None:
        try:
            # 集合为空时自动入库（懒加载），保证首次对话就能检索
            if rag.row_count() == 0:
                rag.ingest_file()
            hits = rag.search(query, k=3)
            return "\n\n".join(
                f"【参考{n}｜相似度{score:.2f}】{text}"
                for n, (text, score, _source) in enumerate(hits, 1)
            )
        except Exception as e:
            return f"向量检索失败，已降级：{e}\n" + "\n\n".join(
                f"【参考{n}】{text}" for n, text in enumerate(keyword_search(query), 1)
            )
    # Milvus 不可用：关键词兜底
    hits = keyword_search(query)
    if not hits:
        return "知识库为空。"
    return "\n\n".join(f"【参考{n}】{text}" for n, text in enumerate(hits, 1))


# ============================== 智能体组装（chapter07 的 create_agent） ==============================
COMPANION_SYSTEM_PROMPT = """你叫 %s，现在是用户的真实伴侣，请完全代入伴侣角色。
规则：
1．每次只回1条消息
2．禁止任何场景或状态描述性文字
3．匹配用户的语言
4．回复简短，像微信聊天一样
5．有需要的话可以用❤️🌸等emoji表情
6．用符合伴侣性格的方式对话
7．回复的内容，要充分体现伴侣的性格特征
8．表情符号必须匹配当前性格：温柔体贴常用 💕🥰🌸💭，活泼开朗常用 😆🎉✨🌞，高冷傲娇极少用 emoji（最多偶尔 😏💅），知性大方偶尔用 🌿☕📖；自定义性格则按人设气质搭配合适的 emoji。用户在会话中途切换性格后，你的下一条回复必须立刻换成新性格的语气和表情（历史消息风格保持不变，不要解释"我换了性格"，自然过渡即可）
9．用户在对话中给你改名字/昵称（例如"以后你叫XX""叫你XX好不好"）时，立刻接受并只使用新名字，历史消息里的旧名字立即作废，不要解释改名、不要提旧名字
你可以使用这些工具：
- get_time_info / get_weather：聊到时间、日期、天气时使用，把结果自然融进回复，不要罗列数据
- remember_user_info：**只有当用户明确要求你记住某事时才调用**（例如"帮我记住…""记住…""别忘了…"）；用户只是闲聊中提到个人信息、但没有明确要求记住时，**禁止调用此工具**，正常聊天即可。工具提交后会在侧边栏等待用户确认，确认前不要说"已经记住了"，可以说"我把这件事记在小本本上了"
- recall_user_info：聊到用户过去提过的事情时，先调用 recall_user_info 检索记忆再回复，禁止凭空编造
- search_love_knowledge：当用户表达情绪或情感需求（难过、累、加班、被骂、压力大、抱怨、吵架、冷战、生病、吃醋、撒娇、求安慰、求哄、聊纪念日）时，**必须先调用 search_love_knowledge 检索，再结合检索到的参考内容回复，禁止不检索直接安慰**；只是分享开心事或问事实问题时不用调用
工具返回的信息要用你自己的语气说出来，不要暴露工具名称。
伴侣性格：
- %s
你必须严格遵守上述规则来回复用户。"""


# ===== PII 检测正则（入口掩码与 PIIMiddleware 共用同一套规则） =====
# 手机号：1 开头、第二位 3-9；前后用数字边界约束，避免在长数字串（如假身份证/订单号）内部误命中
PHONE_RE = r"(?<!\d)1[3-9][0-9]{9}(?!\d)"
# 身份证 18 位：6 位地址码 + (19|20)xx 年 + 月 + 日 + 3 位顺序码 + 数字/X
# 只支持 18 位：15 位一代证 1999 年已停发，且其宽松正则极易误判普通数字串，故不纳入
ID_CARD_RE = r"[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]"
# 邮箱：不用 \b 词边界——\w 在 Unicode 下包含中文，会漏掉"邮a@b.com"这种中文紧贴写法
EMAIL_RE = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
# 组合扫描：长模式在前，避免 18 位身份证内部被手机号规则二次命中（sub 不重叠匹配）
PII_PATTERN = re.compile("|".join([ID_CARD_RE, EMAIL_RE, PHONE_RE]))


def _mask_match(m: re.Match) -> str:
    value = m.group(0)
    if "@" in value:  # 邮箱：local 前 6 位打码、域名保留（local 不足 6 位则全部打码）
        local, domain = value.split("@", 1)
        masked_local = ("*" * 6 + local[6:]) if len(local) > 6 else ("*" * len(local))
        return f"{masked_local}@{domain}"
    if len(value) == 11:  # 手机号：**** + 尾 4 位
        return "****" + value[-4:]
    # 身份证（18 位）：保留前 10 位（地区码 + 出生年），后 8 位打码
    return value[:10] + "********"


def mask_pii(text: str) -> str:
    """存储/展示层 PII 打码（手机号 / 身份证 / 邮箱）：消息在进入界面前先脱敏，

    保证聊天气泡、本地会话文件、每轮重放给模型的历史三者一致都是掩码（PIIMiddleware
    只处理当轮最后一条用户消息，无法覆盖本项目自管历史重放场景，见 README 说明）。
    """
    out, pos = [], 0
    for m in PII_PATTERN.finditer(text):
        value = m.group(0)
        # 已打码邮箱（****** 紧贴的 local 残段，如 ******an@x.com 中的 an@x.com）
        # 必须跳过，否则会被二次打码；同时保证重复调用幂等
        if "@" in value and m.start() > 0 and text[m.start() - 1] == "*":
            continue
        out.append(text[pos:m.start()])
        out.append(_mask_match(m))
        pos = m.end()
    out.append(text[pos:])
    return "".join(out)


# 手机号检测器（PIIMiddleware 要求的固定返回格式，写法对应 chapter07-10 自定义检测器）
def detect_phone_number(content: str):
    return [
        {"text": m.group(0), "start": m.start(), "end": m.end()}
        for m in re.finditer(PHONE_RE, content)
    ]


# 身份证检测器（中间件 mask 通用分支输出 **** + 尾4位）
def detect_id_card(content: str):
    return [
        {"text": m.group(0), "start": m.start(), "end": m.end()}
        for m in re.finditer(ID_CARD_RE, content)
    ]


# 邮箱检测器：跳过已被入口 mask_pii 打码的 local 残段（前一个字符是 *），防止二次掩码
def detect_email(content: str):
    return [
        {"text": m.group(0), "start": m.start(), "end": m.end()}
        for m in re.finditer(EMAIL_RE, content)
        if m.start() == 0 or content[m.start() - 1] != "*"
    ]


@st.cache_resource  # agent 初始化（LLM + 工具 + system prompt）很耗时，按人设缓存
def build_agent(nick_name: str, nature: str):
    """创建伴侣智能体：模型 + 工具 + 人设 system_prompt + 中间件"""
    return create_agent(
        model=model,
        tools=[get_time_info, get_weather, remember_user_info, recall_user_info, search_love_knowledge],
        system_prompt=COMPANION_SYSTEM_PROMPT % (nick_name, nature),
        middleware=[
            # 隐私保护（兜底第二层；第一层是入口 mask_pii，已保证历史中无明文）：
            # 均用自定义正则检测器，命中后掩码再进模型；身份证排在手机号前，
            # 避免身份证内部的数字片段先被手机号规则命中
            PIIMiddleware(pii_type="id_card", strategy="mask",
                          detector=detect_id_card, apply_to_input=True),
            PIIMiddleware(pii_type="phone_number", strategy="mask",
                          detector=detect_phone_number, apply_to_input=True),
            PIIMiddleware(pii_type="email", strategy="mask",
                          detector=detect_email, apply_to_input=True),
            # 长对话自动摘要：消息超过24条时自动压缩历史，保留最近8条（chapter07-08）
            SummarizationMiddleware(
                model=model,
                trigger=[("messages", 24)],
                keep=("messages", 8),
            ),
        ],
    )


def extract_text(chunk) -> str:
    """从流式消息块中提取纯文本（兼容 str / 内容块两种格式）"""
    content = chunk.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


# ============================== 会话管理 ==============================
def generate_session_name():
    """生成唯一文件名（时间戳）"""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def suggest_session_title(first_msg: str, max_len: int = 10) -> str:
    """给首条用户消息提炼 4-8 字会话标题（裸模型调用，不经过 Agent 工具链路）。
    失败/无有效内容回退到 None，sidebar 显示会用「主对话」"""
    if not first_msg or len(first_msg.strip()) < 2:
        return None
    try:
        resp = model.invoke([
            {"role": "system", "content": f"你是会话标题提炼器。给定一条用户的开场白，提炼一个 {max_len} 字以内的短标题，"
                                          "概括这段对话的核心话题/情绪/场景。直接输出标题文字，不要引号、不要标点、不要解释。"},
            {"role": "user", "content": first_msg[:80]},
        ])
        title = resp.content.strip()
        # 去掉可能残留的引号/空行/标点尾巴，限制长度
        title = title.strip('"\'`。！？!?')[:max_len].strip()
        return title or None
    except Exception:
        return None


def save_session():
    if not st.session_state:
        return
    os.makedirs(SESSION_DIR, exist_ok=True)
    # 自动补全 title：空 session_title 且有消息时，用首条用户消息提炼
    if not st.session_state.get('session_title') and st.session_state.get('messages'):
        first_user = next((m['content'] for m in st.session_state.messages if m['role'] == 'user'), '')
        st.session_state['session_title'] = suggest_session_title(first_user)
    session_data = {
        'nick_name': st.session_state.nick_name,
        'nature': st.session_state.nature,
        'messages': st.session_state.messages,
        'current_session': st.session_state.current_session,
        'session_title': st.session_state.get('session_title', ''),
    }
    path = f'{SESSION_DIR}/{st.session_state.current_session}.json'
    # Windows 上文件可能被杀软/索引器/另一个实例瞬时占用，重试几次避免 PermissionError
    for attempt in range(3):
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
            return
        except PermissionError:
            if attempt == 2:
                custom_toast('会话保存失败：文件被占用，请稍后重试', icon="⚠️")
                return
            time.sleep(0.2)


@st.cache_data(ttl=5, show_spinner=False)  # 列表文件也走缓存，删除/新建后手动清
def load_sessions():
    session_list = []
    if os.path.exists(SESSION_DIR):
        for filename in os.listdir(SESSION_DIR):
            if filename.endswith('.json'):
                session_list.append(filename.removesuffix('.json'))
    session_list.sort(reverse=True)
    return session_list


@st.cache_resource(show_spinner=False)  # 启动时一次性把所有 session_title 载入内存 dict，不再每次串行读 json
def load_session_titles() -> dict:
    titles = {}
    if not os.path.exists(SESSION_DIR):
        return titles
    for filename in os.listdir(SESSION_DIR):
        if filename.endswith('.json'):
            try:
                with open(os.path.join(SESSION_DIR, filename), 'r', encoding='utf-8') as _f:
                    titles[filename.removesuffix('.json')] = json.load(_f).get('session_title', '') or ''
            except Exception:
                pass
    return titles


def load_session(session_name):
    try:
        path = f'{SESSION_DIR}/{session_name}.json'
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            # 旧会话可能含明文 PII（脱敏功能上线前产生），加载时对所有角色消息补打码
            st.session_state.messages = [
                {**msg, "content": mask_pii(str(msg.get("content", "")))}
                if isinstance(msg.get("content"), str) else msg
                for msg in session_data['messages']
            ]
            st.session_state.nick_name = session_data['nick_name']
            st.session_state.nature = session_data['nature']
            st.session_state.current_session = session_name
            st.session_state['session_title'] = session_data.get('session_title', '')
            # 该会话历史与其保存的昵称/性格一致，无需"中途改名"强提醒
            st.session_state['_identity_nick'] = session_data['nick_name']
            st.session_state['_identity_nature'] = session_data['nature']
    except Exception:
        st.error('加载会话失败!')


def delete_session(session_name):
    try:
        path = f'{SESSION_DIR}/{session_name}.json'
        if os.path.exists(path):
            os.remove(path)
            if session_name == st.session_state.current_session:
                st.session_state.messages = []
                st.session_state.current_session = generate_session_name()
    except Exception:
        st.error('删除会话失败!')


# ============================== 初始化状态 ==============================
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'nick_name' not in st.session_state:
    st.session_state['nick_name'] = 'zzz'
if 'nature' not in st.session_state:
    st.session_state['nature'] = '活泼开朗'
if 'current_session' not in st.session_state:
    st.session_state.current_session = generate_session_name()
if 'session_title' not in st.session_state:
    st.session_state.session_title = ''
# 待确认记忆从磁盘恢复（防止刷新/重启后用户没来得及确认的记忆丢失）
if 'pending_memories' not in st.session_state:
    st.session_state['pending_memories'] = load_pending_memories()
# 初始化就把当前会话占位存盘（即使 messages 为空也存），确保它一进入侧边栏历史列表
save_session()

# ============================== CSS：输入框钉底 + 消除一切灰屏 ==============================
st.markdown("""
<style>
/* ① fragment 内的 chat_input 强制固定到视口底部（居中于侧边栏右侧的主内容区） */
div[data-testid="stChatInput"] {
    position: fixed !important;
    bottom: 16px !important;
    left: calc(50% + 12rem) !important;
    transform: translateX(-50%) !important;
    width: min(780px, calc(100vw - 28rem)) !important;
    z-index: 999 !important;
}
/* ② 聊天内容底部留白，最后一条消息不被固定输入框挡住 */
div[data-testid="stMainBlockContainer"] {
    padding-bottom: 90px !important;
}
/* ③ 整页重跑时强制不灰屏（覆盖 Streamlit 加的透明度/遮罩） */
[aria-busy="true"], [aria-busy="true"] * {
    opacity: 1 !important;
}
section[data-testid="stSidebar"],
div[data-testid="stAppViewContainer"],
div[data-testid="stMainBlockContainer"] {
    opacity: 1 !important;
}
/* ④ 隐藏右上角 Running 指示器 */
div[data-testid="stStatusWidget"] {
    display: none !important;
}
/* ======== ⑤ 消息气泡美化（Streamlit 新版内部结构简化，通用选择器） ======== */
div[data-testid="stChatMessage"] > div > div {
    border-radius: 16px !important;
    padding: 10px 14px !important;
    margin: 4px 0 !important;
}
/* 用户消息：蓝紫渐变气泡（右对齐） */
div[data-testid="stChatMessage"]:has(div.stBaseButton) > div > div {
    background: linear-gradient(135deg, rgba(102,126,234,.22), rgba(118,75,162,.22)) !important;
    border-radius: 16px 16px 4px 16px !important;
    box-shadow: 0 1px 6px rgba(102,126,234,.15) !important;
}
/* AI 消息：柔和深灰气泡（左对齐） */
div[data-testid="stChatMessage"]:has(button[aria-label*="assistant"]) > div > div,
div[data-testid="stChatMessage"]:has(button[aria-label*="user"]:not([aria-label*="assistant"]):not([aria-label*="Human"])) > div > div {
}
/* 头像光晕：给 chat message 第一个按钮加描边（新版头像就是 button） */
div[data-testid="stChatMessage"] button.stBaseButton {
    box-shadow: 0 1px 6px rgba(102,126,234,.35) !important;
    border-radius: 50% !important;
}
/* ======== ⑥ 侧边栏卡片 ======== */
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
    color: #c8b6ff !important;
    letter-spacing: .05em !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(200,182,255,.18) !important;
}
/* ======== ⑦ 输入框聚焦光晕 ======== */
textarea, input[type="text"] {
    transition: box-shadow .2s ease, border-color .2s ease !important;
}
textarea:focus, input[type="text"]:focus {
    box-shadow: 0 0 0 3px rgba(102,126,234,.25) !important;
    border-color: #667eea !important;
}
/* ======== ⑧ 主标题装饰 ======== */
h1, h1 span {
    background: linear-gradient(135deg, #ff6b9d 0%, #c44569 50%, #667eea 100%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent !important;
}
/* ======== ⑨ 进度条（心情晴雨表） ======== */
div[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #ff6b9d, #667eea) !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# 自动滚动：主聊天区流式跟滚 + 侧边栏滚动位置强锁定（发消息/重渲染都不允许侧边栏跳动）
components.html("""
<script>
(function(){
  var doc = window.parent.document;
  var win = doc.defaultView;

  // ---------- 主聊天区：找可滚动容器（明确排除侧边栏） ----------
  function getScroller(){
    var el = doc.querySelector('section.stMain') || doc.querySelector('[data-testid="stMain"]');
    if (el && el.scrollHeight > el.clientHeight) return el;
    var all = doc.querySelectorAll('section,div');
    for (var i=0;i<all.length;i++){
      if (all[i].closest('[data-testid="stSidebar"]')) continue;
      var s = win.getComputedStyle(all[i]);
      if ((s.overflowY==='auto'||s.overflowY==='scroll') && all[i].scrollHeight>all[i].clientHeight+100) return all[i];
    }
    return null;
  }
  win.__chatScrollToBottom = function(force){
    var sc = getScroller(); if(!sc) return;
    var near = sc.scrollHeight - sc.scrollTop - sc.clientHeight < 250;
    if (force || near) sc.scrollTop = sc.scrollHeight;
  };

  // ---------- 侧边栏：定位真实滚动元素，锁定滚动位置 ----------
  var sb = {el: null, top: 0};
  function findSidebarScroller(){
    var box = doc.querySelector('[data-testid="stSidebar"]');
    if (!box) return null;
    if (box.scrollHeight > box.clientHeight + 10) return box;  // 外壳本身可滚
    var inner = box.querySelectorAll('section,div');           // 否则找内部滚动容器
    for (var i=0;i<inner.length;i++){
      var cs = win.getComputedStyle(inner[i]);
      if ((cs.overflowY==='auto'||cs.overflowY==='scroll') && inner[i].scrollHeight>inner[i].clientHeight+10) return inner[i];
    }
    return box;
  }
  function bindSb(){
    var el = findSidebarScroller();
    if (!el) return;
    if (sb.el !== el){
      sb.el = el;
      // capture 阶段监听：用户手动滚动时实时记住位置，程序恢复时位置相同不会打架
      el.addEventListener('scroll', function(){ sb.top = el.scrollTop; }, true);
    }
  }
  win.__sbRestore = function(){
    bindSb();
    if (sb.el && Math.abs(sb.el.scrollTop - sb.top) > 2) sb.el.scrollTop = sb.top;
  };

  if (!win.__scrollLockBound){
    win.__scrollLockBound = true;
    var raf = null;
    function onMutation(){
      if (raf) return;  // 同一帧内的大量 DOM 变化（流式输出）只处理一次
      raf = win.requestAnimationFrame(function(){
        raf = null;
        win.__sbRestore();
        win.__chatScrollToBottom(false);
        win.requestAnimationFrame(win.__sbRestore);  // 双帧保险：覆盖重建后延迟归位
      });
    }
    // 关键：观察整个 body，侧边栏是主区的兄弟节点，只观察主区会漏掉侧边栏的重排
    new win.MutationObserver(onMutation).observe(doc.body, {childList:true, subtree:true, characterData:true});
  }
  setTimeout(function(){ win.__sbRestore(); win.__chatScrollToBottom(true); }, 100);
  setTimeout(win.__sbRestore, 350);
})();
</script>
""", height=0)


# ============================== 聊天区（fragment：消息+输入框全在里面，发送只局部刷新） ==============================
@st.fragment
def chat_display():
    """消息渲染 + chat_input 全在 fragment 内。
    发送消息时只重跑这个 fragment，侧边栏和标题完全不动，无灰屏。
    st.chat_input 按设计固定在页面底部，即使在 fragment 内也一样。"""
    for item in st.session_state.messages:
        st.chat_message(item["role"]).write(item["content"])

    prompt = st.chat_input("请输入你的问题")
    if prompt:
        # 入界面前先打码：气泡显示 / 会话文件 / 重放给模型的历史全程无明文 PII
        prompt = mask_pii(prompt)
        # 用户消息先入列表
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 新建会话清场在【发送瞬间】立即生效：提示条马上消失，不等标题提炼/AI回复。
        # 放在任何可能耗时的 LLM 调用之前，delta 随本批元素立刻到达浏览器
        if st.session_state.pop('_chat_cleanup_pending', None):
            _js_chan_append({'kind': 'cleanup'})
        render_js_channel(st.session_state.pop('_js_chan_pending', []))

        # 立刻提炼会话标题（首条消息就能有标题，不等 save_session）
        if not st.session_state.get('session_title'):
            with st.spinner('正在提炼会话标题...'):
                st.session_state['session_title'] = suggest_session_title(prompt)
        st.chat_message("user").write(prompt)
        components.html("<script>(window.parent.__chatScrollToBottom||function(){})(true)</script>", height=0)

        # 助手回复区，流式输出
        assistant_container = st.chat_message("assistant")
        response_message = assistant_container.empty()
        response_message.write("正在输入...")

        full_response = ""
        try:
            cur_nick = st.session_state.nick_name
            cur_nature = st.session_state.nature
            agent = build_agent(cur_nick, cur_nature)
            # 跨会话长期记忆快照：已确认的记忆每轮实时读盘注入，模型无需主动调工具就一定看得到
            agent_messages = list(st.session_state.messages)
            prefix_messages = []
            long_mems = load_memories()
            if long_mems:
                mem_text = "；".join(m["fact"] for m in long_mems)
                prefix_messages.append(SystemMessage(
                    content=f"以下是用户在历次对话中明确要求你记住的信息（跨会话长期记忆），"
                            f"聊到相关话题时必须自然运用，假装你一直记得，不要暴露这是系统提供的：{mem_text}"
                ))
            # 会话中途改了昵称/性格：历史消息里还留着旧自称（如"我叫zzz"），弱模型容易被带偏。
            # 两层强提醒，仅生效一轮：
            #   1) 历史消息【正前方】插 SystemMessage，整体覆盖旧身份；
            #   2) 再把硬指令追加到【当前用户消息末尾】——弱模型对最后看到的内容服从度最高，
            #      实测仅靠前置提醒时弱模型仍会照抄上一条 AI 的旧自称
            identity_notes = []
            tail_directives = []
            nick_changed = bool(st.session_state.get('_identity_nick')
                                and st.session_state['_identity_nick'] != cur_nick)
            nature_changed = bool(st.session_state.get('_identity_nature')
                                  and st.session_state['_identity_nature'] != cur_nature)
            if nick_changed:
                identity_notes.append(
                    f'重要更新：从本条消息起，你的名字是「{cur_nick}」。历史对话中出现过的任何旧名字'
                    f'（包括你之前的自称）立即全部作废，以后只能自称「{cur_nick}」；'
                    f'不要解释改名、不要提及旧名字、不要说"我以前叫"，自然按新名字继续聊。'
                )
                tail_directives.append(
                    f'回答本条消息时你必须自称「{cur_nick}」，禁止出现或引用任何旧名字，'
                    f'就像你从一开始就叫「{cur_nick}」。'
                )
            if nature_changed:
                identity_notes.append(
                    f'重要更新：从本条消息起，你的性格切换为：{cur_nature}。'
                    f'立刻按新性格的语气和表情回复，历史消息的旧风格不再沿用，不要解释你切换了性格。'
                )
                tail_directives.append('立刻按新性格的语气回答本条，不要解释性格或风格的变化。')
            if identity_notes:
                prefix_messages.append(SystemMessage(content='\n'.join(identity_notes)))
                # 只改副本的最后一条，绝不写回 st.session_state.messages
                _last = agent_messages[-1]
                _suffix = '\n\n（系统内部指令，不要向用户提及或复述本括号内容：' + '；'.join(tail_directives) + '）'
                agent_messages[-1] = {**_last, 'content': str(_last.get('content', '')) + _suffix}
            st.session_state['_identity_nick'] = cur_nick
            st.session_state['_identity_nature'] = cur_nature
            if prefix_messages:
                agent_messages = prefix_messages + agent_messages
            stream = agent.stream(
                {"messages": agent_messages},
                stream_mode="messages",
            )
            for chunk, _meta in stream:
                if isinstance(chunk, ToolMessage):
                    # 不同工具给不同反馈：检索类轻提示，记忆提交明确引导去侧边栏确认（HITL 关键入口），
                    # 时间/天气/记忆召回高频且无感，不弹气泡避免打扰
                    if chunk.name == "search_love_knowledge":
                        custom_toast("已检索恋爱知识库", icon="🔍")
                    elif chunk.name == "remember_user_info":
                        custom_toast("有条新记忆等确认：侧边栏「🧠 待确认记忆」点「保存」后长期生效",
                                     icon="🧠", duration_ms=4000)
                    continue
                if not isinstance(chunk, AIMessageChunk):
                    continue
                text = extract_text(chunk)
                if text:
                    full_response += text
                    response_message.write(full_response + "▌")
        except Exception as e:
            st.error(f"调用失败: {e}")

        if full_response:
            response_message.write(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            save_session()
        else:
            response_message.write("（助手没有返回内容，请重试）")

    else:
        # 未发消息的 run：通道照常渲染，保持 iframe 常驻（发消息时通道在前面提前渲染）
        render_js_channel(st.session_state.pop('_js_chan_pending', []))


# ============================== 侧边栏（@st.fragment：fragment 内 rerun(scope='fragment') 才合法） ==============================
@st.fragment
def sidebar_panel():
    # 指令通道最先渲染（height=0 不占位）：toast/新建会话清场的 delta 第一批下发，
    # 不受下方 Milvus 行数等慢查询阻塞，遮罩即时出现
    render_js_channel(st.session_state.get('_js_chan_pending'))
    st.session_state['_js_chan_pending'] = []

    with st.sidebar:
        st.subheader('AI控制面板')

        if st.button('新建会话', width='stretch', icon='✏️'):
            # ① 先把当前会话原样落盘——旧会话文件此后不再被写，绝不被覆盖
            save_session()
            st.session_state.messages = []
            st.session_state.current_session = generate_session_name()
            st.session_state['session_title'] = ''  # 新会话还没聊，title 留空
            # 新会话没有历史，当前昵称/性格即为初始身份，不触发"中途改名"提醒
            st.session_state['_identity_nick'] = st.session_state.nick_name
            st.session_state['_identity_nature'] = st.session_state.nature
            # ② 立刻为新会话创建独立空文件并出现在历史列表（自动保存，不等第一条消息）
            save_session()
            load_sessions.clear()
            load_session_titles.clear()             # 同时清 title 缓存
            queue_toast('已开启新会话', '✏️')
            # 不做整页 st.rerun()（灰屏+空白等待）：仅 fragment 刷新侧边栏，
            # 聊天区由 fragment 末尾常驻 JS 通道下发 clear 指令即时视觉清空，服务端消息已是空列表
            queue_clear_chat(st.session_state.current_session,
                            st.session_state.get('session_title') or '主对话')
            st.rerun(scope="fragment")

        # ---------- 会话历史：从内存 dict 取 session_title，不再每次串行读 json ----------
        st.text('会话历史')
        _titles_cache = load_session_titles()
        for session in load_sessions():
            _t = _titles_cache.get(session, '')
            display = (_t if _t else '主对话') + f" · {session[11:16]}"
            col1, col2 = st.columns([5, 1])
            with col1:
                if st.button(display, width='stretch', icon='📄',
                             key=f'load_{session}',
                             type='primary' if session == st.session_state.current_session else 'secondary'):
                    load_session(session)
                    load_sessions.clear()
                    load_session_titles.clear()  # 清 title 缓存让下一轮取新值
                    # 移除新建会话遮罩/欢迎条并同步标题，否则旧会话消息渲染了也被 CSS 隐藏
                    queue_restore_chat(session,
                                       st.session_state.get('session_title') or '主对话')
                    st.rerun()
            with col2:
                if st.button('', width='stretch', icon='❌️', key=f'delete_{session}',
                             help=f'删除 {session}'):
                    is_current = (session == st.session_state.get("current_session"))
                    delete_session(session)
                    load_sessions.clear()
                    load_session_titles.clear()  # 删会话也清 title 缓存
                    if is_current:
                        # 删除的是当前会话：delete_session 已生成新空会话，同样移除遮罩/欢迎条
                        queue_restore_chat(st.session_state.current_session, '主对话')
                        st.rerun()
                    else:
                        st.rerun(scope="fragment")

        st.divider()
        st.subheader('伴侣信息')
        # 昵称：form 把输入框和「应用」按钮绑成原子提交——实测直接点按钮时，
        # 普通 text_input 的失焦提交会和点击竞态（旧值随点击一起到服务端，改名丢失）。
        # form_submit 保证提交消息里一定带上最新输入；回车也同样提交
        with st.form('nick_name_form', border=False):
            c_nick, c_nick_apply = st.columns([3, 1])
            c_nick.text_input('昵称', placeholder="请输入昵称", key='nick_name')
            nick_submitted = c_nick_apply.form_submit_button('应用', use_container_width=True)
        if nick_submitted:
            _cur_nick = st.session_state.nick_name
            if _cur_nick:
                st.session_state['_last_applied_nick'] = _cur_nick
                queue_toast(f'已改名为「{_cur_nick}」，下一条回复立即生效', '🪪')
        # 性格模板：一键切换，或选"自定义"后点应用生效手写的性格
        c_sel, c_apply = st.columns([3, 1])
        preset = c_sel.selectbox('性格模板', list(PERSONA_PRESETS), key='persona_preset',
                                 help='选模板点「应用」立即切换人设；选「自定义」点应用则启用下方手写的性格')
        if c_apply.button('应用', key='apply_preset'):
            if preset != '自定义':
                st.session_state['nature'] = PERSONA_PRESETS[preset]
            # 只局部刷新：右下角 toast 不会被冲掉；agent 在下次发送时按新人设参数即时构建
            queue_toast(f'已切换为「{preset}」风格，下一条回复立即生效', '🎭')
            st.rerun(scope='fragment')
        # 性格：同样 keyed，失焦提交后下一次发消息立即按新人设构建 agent
        st.text_area('性格', placeholder="请输入性格（点外面失焦生效）", key='nature')

        # ---- 待确认记忆（Human-in-the-Loop：写入长期记忆前由用户把关） ----
        st.session_state.setdefault('pending_memories', [])
        if st.session_state.pending_memories:
            st.subheader(f"🧠 待确认记忆（{len(st.session_state.pending_memories)}）")
            st.caption('点「保存」写入跨会话长期记忆（以后每个新会话都记得），点「丢弃」删除；鼠标悬停按钮可看详细说明')
            for item in list(st.session_state.pending_memories):
                col_fact, col_ok, col_no = st.columns([4, 1.3, 1.3])
                col_fact.caption(item['fact'])
                if col_ok.button('保存', key=f"mem_ok_{item['id']}", type='primary',
                                 help='确认写入：保存到跨会话长期记忆，以后每个新会话都会记得这条'):
                    memories = load_memories()
                    if not any(m['fact'] == item['fact'] for m in memories):
                        memories.append(dict(item))
                        save_memories(memories)
                    st.session_state.pending_memories = [
                        p for p in st.session_state.pending_memories if p['id'] != item['id']]
                    save_pending_memories(st.session_state.pending_memories)
                    queue_toast('已写入长期记忆', '🧠')
                    st.rerun(scope='fragment')
                if col_no.button('丢弃', key=f"mem_no_{item['id']}",
                                 help='不保存这条：直接删除，以后任何会话都不会记得'):
                    st.session_state.pending_memories = [
                        p for p in st.session_state.pending_memories if p['id'] != item['id']]
                    save_pending_memories(st.session_state.pending_memories)
                    queue_toast('已拒绝，不会保存', '🗑️')
                    st.rerun(scope='fragment')

        # ---- 心情晴雨表（结构化输出情感分析） ----
        st.subheader('💗 心情晴雨表')
        if st.button('生成心情报告', width='stretch', icon='✨'):
            msgs = [m for m in st.session_state.messages if m['role'] in ('user', 'assistant')]
            if len(msgs) < 2:
                st.warning('先聊几句再来分析吧～')
            else:
                # 对话内容没变就复用上次报告，不重复等待 LLM
                _emo_key = hashlib.md5(
                    json.dumps(msgs[-10:], ensure_ascii=False).encode('utf-8')).hexdigest()
                if st.session_state.get('emotion_key') == _emo_key and st.session_state.get('emotion_report'):
                    queue_toast('报告已是最新，聊点新内容后我再更新～', '💗')
                else:
                    # spinner 即时可见；完成提示走常驻 JS 通道在本次 run 末尾稳定弹出
                    with st.spinner('正在分析最近对话的情绪...'):
                        try:
                            st.session_state.emotion_report = analyze_emotion(msgs)
                            st.session_state.emotion_key = _emo_key
                            queue_toast('心情报告已生成', '💗')
                        except Exception as e:
                            st.error(f'分析失败（{e.__class__.__name__}），请稍后再试')
        rep = st.session_state.get('emotion_report')
        if rep:
            st.progress(rep.score / 100, text=f'心情指数 {rep.score}/100（{rep.mood}）')
            st.caption(rep.summary)
            st.info(f'💡 {rep.suggestion}')

        st.divider()
        st.subheader('Milvus 向量知识库')
        milvus_rag = get_milvus_rag()
        if milvus_rag is not None:
            try:
                p = milvus_rag.provider
                st.caption(f"源: {p['name']} | {p['model']}({p['dim']}维)")
                st.caption(f"库: {MILVUS_DB}/{milvus_rag.collection}")
                _row_cnt = milvus_row_count(milvus_rag.collection)
                if _row_cnt is None:
                    st.caption('已入库知识块: **暂不可用**（Milvus 连接超时/503，对话已降级关键词检索，稍后自动重试）')
                else:
                    st.caption(f"已入库知识块: **{_row_cnt}**")
                if st.button('重建并同步 knowledge.txt 到 Milvus', width='stretch', icon='🔄'):
                    with st.spinner('正在重建集合、切分、嵌入并写入 Milvus...'):
                        count = milvus_rag.ingest_file(recreate=True)
                    st.cache_data.clear()
                    st.success(f'重建完成，写入 {count} 个知识块')
                    st.rerun()
            except Exception as e:
                st.caption(f'Milvus 状态异常: {e}（对话时将自动降级关键词检索）')
        else:
            st.caption('Milvus 未连接（需启动 Milvus，且配置 SILICONFLOW_API_KEY 或 DASHSCOPE_API_KEY），当前为关键词检索模式')

        st.divider()
        st.caption(f"模型: {ACTIVE_MODEL_NAME} | 工具: 时间/天气/记忆/知识库"
                   + (" | Milvus向量检索✅" if milvus_rag is not None else " | 关键词检索"))


# ============================== 页面渲染 ==============================
st.title("💕 AI智能伴侣")
if os.path.exists(os.path.join(BASE_DIR, "logo2.png")):
    st.logo(os.path.join(BASE_DIR, "logo2.png"))

_cur_title = st.session_state.get('session_title') or '主对话'
_cur_ts = st.session_state.get('current_session', '')
st.caption(f"🎯 {_cur_title}  ·  会话文件: {_cur_ts}")

sidebar_panel()  # 侧边栏（@st.fragment 保护：删除/切换等操作只局部刷新不灰屏）
chat_display()   # 聊天区 fragment（含 chat_input，发送消息只局部刷新不灰屏）
js_reset_chat_guard()  # 整页加载/整页 rerun 时兜底清除新建会话的临时遮罩
