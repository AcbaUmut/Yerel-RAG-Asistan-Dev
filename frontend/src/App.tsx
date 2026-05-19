import { useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./App.css";

type Doc = {
  file_name: string;
  chunk_count: number;
  section_count: number;
  added_at: string;
};

type Message = {
  role: "user" | "assistant";
  content: string;
};

type ToastMsg = {
  type: "success" | "error" | "info";
  message: string;
};

// <think>...</think> bloklarını metin parçalarından ayırır.
// Akış sırasında blok kapanmamış olabilir; o zaman 'complete: false' işaretlenir.
function parseThinkBlocks(content: string) {
  const regex = /<think>([\s\S]*?)(?:<\/think>|$)/g;
  const parts: Array<{
    type: "think" | "text";
    content: string;
    complete?: boolean;
  }> = [];
  let lastEnd = 0;
  let match: RegExpExecArray | null;
  while ((match = regex.exec(content)) !== null) {
    if (match.index > lastEnd) {
      parts.push({
        type: "text",
        content: content.slice(lastEnd, match.index),
      });
    }
    const fullMatch = content.slice(match.index, match.index + match[0].length);
    parts.push({
      type: "think",
      content: match[1],
      complete: fullMatch.endsWith("</think>"),
    });
    lastEnd = match.index + match[0].length;
  }
  if (lastEnd < content.length) {
    parts.push({ type: "text", content: content.slice(lastEnd) });
  }
  return parts;
}

// Asistan mesajını markdown olarak render eder; <think> bloklarını
// açılır-kapanır olarak gösterir.
function AssistantMessage({ content }: { content: string }) {
  const parts = parseThinkBlocks(content);
  return (
    <div className="space-y-2">
      {parts.map((part, i) => {
        if (part.type === "think") {
          return (
            <details key={i} className="text-xs text-gray-500">
              <summary className="cursor-pointer hover:text-gray-700 select-none">
                {part.complete ? "Düşünme sürecini gör" : "Düşünüyor..."}
              </summary>
              <div className="mt-1 pl-3 border-l-2 border-gray-300 whitespace-pre-wrap text-gray-600">
                {part.content.trim()}
              </div>
            </details>
          );
        }
        return (
          <div key={i} className="prose prose-sm max-w-none">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {part.content}
            </ReactMarkdown>
          </div>
        );
      })}
    </div>
  );
}

function App() {
  const [docs, setDocs] = useState<Doc[]>([]);
  const [collection, setCollection] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedDoc, setSelectedDoc] = useState<string | null>(null);

  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState<string>("");
  const [isStreaming, setIsStreaming] = useState<boolean>(false);

  const [isUploading, setIsUploading] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [pendingFiles, setPendingFiles] = useState<File[]>([]);
  const [conflicts, setConflicts] = useState<string[]>([]);
  const [decisions, setDecisions] = useState<
    Record<string, "overwrite" | "skip">
  >({});

  const [deletingDoc, setDeletingDoc] = useState<string | null>(null);

  const [allCollections, setAllCollections] = useState<string[]>([]);
  const [showCollectionMenu, setShowCollectionMenu] = useState<boolean>(false);
  const collectionMenuRef = useRef<HTMLDivElement>(null);

  const [maxFileSizeMb, setMaxFileSizeMb] = useState<number>(50);
  const [toast, setToast] = useState<ToastMsg | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState<boolean>(true);

  useEffect(() => {
    Promise.all([
      fetch("http://localhost:8000/documents").then((r) => r.json()),
      fetch("http://localhost:8000/collections").then((r) => r.json()),
      fetch("http://localhost:8000/health").then((r) => r.json()),
    ])
      .then(([docsData, colData, healthData]) => {
        setDocs(docsData.documents);
        setCollection(docsData.collection);
        setAllCollections(colData.all);
        if (typeof healthData.max_file_size_mb === "number") {
          setMaxFileSizeMb(healthData.max_file_size_mb);
        }
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  // Dropdown dışına tıklayınca kapansın
  useEffect(() => {
    if (!showCollectionMenu) return;
    function onMouseDown(e: MouseEvent) {
      if (
        collectionMenuRef.current &&
        !collectionMenuRef.current.contains(e.target as Node)
      ) {
        setShowCollectionMenu(false);
      }
    }
    document.addEventListener("mousedown", onMouseDown);
    return () => document.removeEventListener("mousedown", onMouseDown);
  }, [showCollectionMenu]);

  // Toast'u 4 saniye sonra otomatik kapat
  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), 4000);
    return () => clearTimeout(t);
  }, [toast]);

  // Mesaj geldikçe otomatik en alta kay — ama kullanıcı yukarı kaydırdıysa karma.
  useEffect(() => {
    if (autoScroll) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, autoScroll]);

  // Kullanıcı scroll edince autoScroll'u güncelle.
  // En altta mı? → autoScroll açık. Yukarıdaysa → kapalı.
  function handleMessagesScroll() {
    const el = messagesContainerRef.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 30;
    setAutoScroll(atBottom);
  }

  async function handleSend() {
    if (!input.trim() || !selectedDoc || isStreaming) return;

    const question = input.trim();
    setInput("");
    setIsStreaming(true);

    setMessages((prev) => [
      ...prev,
      { role: "user", content: question },
      { role: "assistant", content: "" },
    ]);

    try {
      const response = await fetch("http://localhost:8000/query/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, file_name: selectedDoc }),
      });

      if (response.status === 409) {
        // Sunucu meşgul — başka sorgu işleniyor. Mesajı silip toast göster.
        setMessages((prev) => prev.slice(0, -2)); // user + boş asistanı sil
        setToast({
          type: "info",
          message:
            "Sunucu şu an başka bir sorgu işliyor. Birkaç saniye bekleyip tekrar deneyin.",
        });
        return;
      }

      if (!response.ok || !response.body) {
        throw new Error(`HTTP ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });

        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = {
            ...updated[updated.length - 1],
            content: updated[updated.length - 1].content + chunk,
          };
          return updated;
        });
      }
    } catch (err) {
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: "assistant",
          content: `[Hata: ${err instanceof Error ? err.message : "bilinmeyen"}]`,
        };
        return updated;
      });
    } finally {
      setIsStreaming(false);
    }
  }

  async function refreshDocs() {
    try {
      const res = await fetch("http://localhost:8000/documents");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setDocs(data.documents);
    } catch (err) {
      console.error("Liste yenilenemedi:", err);
    }
  }

  async function handleDeleteDoc(fileName: string) {
    if (!confirm(`'${fileName}' silinecek. Emin misin?`)) return;

    setDeletingDoc(fileName);
    try {
      const res = await fetch("http://localhost:8000/documents", {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ file_names: [fileName] }),
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const result = await res.json();
      if (result.failed && result.failed.length > 0) {
        alert(`Silinemedi: ${result.failed[0].reason}`);
      } else {
        // Eğer silinen dosya seçiliyse seçimi kaldır
        if (selectedDoc === fileName) {
          setSelectedDoc(null);
          setMessages([]);
        }
        await refreshDocs();
      }
    } catch (err) {
      alert(
        `Silme hatası: ${err instanceof Error ? err.message : "bilinmeyen"}`,
      );
    } finally {
      setDeletingDoc(null);
    }
  }

  async function handleSwitchCollection(name: string) {
    if (name === collection) {
      setShowCollectionMenu(false);
      return;
    }

    try {
      const res = await fetch(
        `http://localhost:8000/collections/${name}/activate`,
        { method: "POST" },
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      // Yeni koleksiyondaki dokümanları çek
      const docsRes = await fetch("http://localhost:8000/documents");
      const docsData = await docsRes.json();

      setCollection(name);
      setDocs(docsData.documents);
      setSelectedDoc(null);
      setMessages([]);
      setShowCollectionMenu(false);
    } catch (err) {
      alert(
        `Koleksiyon değiştirilemedi: ${err instanceof Error ? err.message : "bilinmeyen"}`,
      );
    }
  }

  async function handleCreateCollection() {
    const name = prompt("Yeni koleksiyon adı:");
    if (!name || !name.trim()) return;

    try {
      const res = await fetch("http://localhost:8000/collections", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: name.trim() }),
      });

      if (!res.ok) {
        const errBody = await res.json().catch(() => null);
        throw new Error(errBody?.detail || `HTTP ${res.status}`);
      }

      const data = await res.json();
      setAllCollections((prev) => [...prev, data.created]);
      setShowCollectionMenu(false);
    } catch (err) {
      alert(
        `Koleksiyon oluşturulamadı: ${err instanceof Error ? err.message : "bilinmeyen"}`,
      );
    }
  }

  async function handleDeleteCollection(name: string) {
    if (
      !confirm(
        `'${name}' koleksiyonu ve içindeki tüm dokümanlar silinecek. Emin misin?`,
      )
    )
      return;

    try {
      const res = await fetch(`http://localhost:8000/collections/${name}`, {
        method: "DELETE",
      });

      if (!res.ok) {
        const errBody = await res.json().catch(() => null);
        throw new Error(errBody?.detail || `HTTP ${res.status}`);
      }

      // Eğer aktif koleksiyon silindiyse default'a düştük —
      // backend otomatik yaptı, frontend'i de senkronla.
      const wasActive = name === collection;
      setAllCollections((prev) => prev.filter((c) => c !== name));

      if (wasActive) {
        const docsRes = await fetch("http://localhost:8000/documents");
        const docsData = await docsRes.json();
        setCollection(docsData.collection);
        setDocs(docsData.documents);
        setSelectedDoc(null);
        setMessages([]);
      }
    } catch (err) {
      alert(
        `Koleksiyon silinemedi: ${err instanceof Error ? err.message : "bilinmeyen"}`,
      );
    }
  }

  async function handleUpload(files: FileList) {
    if (files.length === 0) return;

    // Boyut ön kontrolü — backend de yapacak ama kullanıcıyı erken uyaralım
    const allFiles = Array.from(files);
    const tooBig: string[] = [];
    const okFiles: File[] = [];
    for (const f of allFiles) {
      const sizeMb = f.size / (1024 * 1024);
      if (sizeMb > maxFileSizeMb) {
        tooBig.push(`${f.name} (${sizeMb.toFixed(1)} MB)`);
      } else {
        okFiles.push(f);
      }
    }

    if (tooBig.length > 0) {
      setToast({
        type: "error",
        message: `Boyut limiti aşıldı (${maxFileSizeMb} MB): ${tooBig.join(", ")}`,
      });
    }

    if (okFiles.length === 0) {
      if (fileInputRef.current) fileInputRef.current.value = "";
      return;
    }

    const fileArr = okFiles;

    try {
      const checkRes = await fetch("http://localhost:8000/documents/check", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ file_names: fileArr.map((f) => f.name) }),
      });

      if (!checkRes.ok) throw new Error(`HTTP ${checkRes.status}`);

      const checkData = await checkRes.json();

      if (checkData.existing.length === 0) {
        // Çakışma yok, direkt yükle
        await doUpload(fileArr, {});
      } else {
        // Çakışma var, modal aç
        setPendingFiles(fileArr);
        setConflicts(checkData.existing);
        // Varsayılan: hepsi skip
        const initial: Record<string, "overwrite" | "skip"> = {};
        for (const name of checkData.existing) {
          initial[name] = "skip";
        }
        setDecisions(initial);
      }
    } catch (err) {
      alert(
        `Kontrol hatası: ${err instanceof Error ? err.message : "bilinmeyen"}`,
      );
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  async function doUpload(
    files: File[],
    decisionsMap: Record<string, "overwrite" | "skip">,
  ) {
    // Çakışma modal'ı hemen kapansın — yükleme başlıyor
    setPendingFiles([]);
    setConflicts([]);
    setDecisions({});

    setIsUploading(true);
    try {
      const formData = new FormData();
      for (const file of files) {
        formData.append("files", file);
      }
      formData.append("decisions", JSON.stringify(decisionsMap));

      const response = await fetch("http://localhost:8000/documents", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const result = await response.json();
      console.log("Yükleme sonucu:", result);

      // Toast özeti
      const successCount = result.success?.length ?? 0;
      const skippedCount = result.skipped?.length ?? 0;
      const failedCount = result.failed?.length ?? 0;
      const summary: string[] = [];
      if (successCount > 0) summary.push(`${successCount} yüklendi`);
      if (skippedCount > 0) summary.push(`${skippedCount} atlandı`);
      if (failedCount > 0) summary.push(`${failedCount} başarısız`);
      setToast({
        type: failedCount > 0 ? "error" : "success",
        message: summary.join(" · ") || "Hiçbir dosya işlenmedi",
      });

      await refreshDocs();
    } catch (err) {
      console.error("Yükleme hatası:", err);
      setToast({
        type: "error",
        message: `Yükleme hatası: ${err instanceof Error ? err.message : "bilinmeyen"}`,
      });
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  }

  function confirmConflicts() {
    doUpload(pendingFiles, decisions);
  }

  function cancelConflicts() {
    setPendingFiles([]);
    setConflicts([]);
    setDecisions({});
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  function setAllDecisions(value: "overwrite" | "skip") {
    const next: Record<string, "overwrite" | "skip"> = {};
    for (const name of conflicts) {
      next[name] = value;
    }
    setDecisions(next);
  }

  return (
    <div className="flex flex-col h-screen">
      <header className="border-b px-4 py-2 flex items-center gap-4 relative">
        <h1 className="text-lg font-semibold">Yerel RAG Asistanı</h1>
        {selectedDoc && (
          <span className="text-sm text-gray-700">
            Seçili: <span className="font-medium">{selectedDoc}</span>
          </span>
        )}
        <div className="ml-auto relative" ref={collectionMenuRef}>
          <button
            onClick={() => setShowCollectionMenu((v) => !v)}
            className="text-sm text-gray-700 hover:bg-gray-100 px-3 py-1 rounded flex items-center gap-1"
          >
            Koleksiyon: <span className="font-medium">{collection || "—"}</span>
            <span className="text-xs">▼</span>
          </button>

          {showCollectionMenu && (
            <div className="absolute right-0 top-full mt-1 bg-white border rounded shadow-lg min-w-50 z-50">
              <ul className="py-1">
                {allCollections.map((name) => (
                  <li
                    key={name}
                    className={`group flex items-center justify-between gap-2 px-3 py-1 text-sm cursor-pointer hover:bg-gray-100 ${
                      name === collection ? "font-medium text-blue-600" : ""
                    }`}
                    onClick={() => handleSwitchCollection(name)}
                  >
                    <span className="truncate">{name}</span>
                    {name !== "default" && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDeleteCollection(name);
                        }}
                        className="opacity-0 group-hover:opacity-100 text-gray-500 hover:text-red-600 text-xs px-1"
                        title="Koleksiyonu sil"
                      >
                        ×
                      </button>
                    )}
                  </li>
                ))}
                <li
                  onClick={handleCreateCollection}
                  className="border-t mt-1 px-3 py-1 text-sm cursor-pointer hover:bg-gray-100 text-blue-600"
                >
                  + Yeni Koleksiyon
                </li>
              </ul>
            </div>
          )}
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">
        <aside className="w-64 border-r p-4 overflow-y-auto">
          <div className="flex items-center justify-between mb-2">
            <h2 className="font-semibold">Dokümanlar</h2>
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={isUploading}
              className="text-xs bg-blue-500 text-white px-2 py-1 rounded hover:bg-blue-600 disabled:bg-gray-300"
            >
              {isUploading ? "..." : "+ Ekle"}
            </button>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf"
            multiple
            className="hidden"
            onChange={(e) => {
              if (e.target.files) handleUpload(e.target.files);
            }}
          />

          {loading && <p className="text-sm text-gray-500">Yükleniyor...</p>}
          {error && <p className="text-sm text-red-600">Hata: {error}</p>}
          {!loading && !error && docs.length === 0 && (
            <p className="text-sm text-gray-500">Doküman yok.</p>
          )}
          {!loading && !error && docs.length > 0 && (
            <ul className="space-y-1 text-sm">
              {docs.map((doc) => (
                <li
                  key={doc.file_name}
                  className={`group flex items-center justify-between gap-2 px-2 py-1 rounded cursor-pointer ${
                    selectedDoc === doc.file_name
                      ? "bg-blue-100 text-blue-900"
                      : "hover:bg-gray-100"
                  }`}
                  onClick={() =>
                    setSelectedDoc((prev) =>
                      prev === doc.file_name ? null : doc.file_name
                    )
                  }
                >
                  <span className="truncate">{doc.file_name}</span>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDeleteDoc(doc.file_name);
                    }}
                    disabled={deletingDoc === doc.file_name}
                    className="opacity-0 group-hover:opacity-100 text-gray-500 hover:text-red-600 text-xs px-1 disabled:opacity-50"
                    title="Sil"
                  >
                    {deletingDoc === doc.file_name ? "..." : "×"}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </aside>

        <main className="flex-1 flex flex-col">
          {messages.length > 0 && (
            <div className="border-b px-4 py-1 flex items-center justify-end">
              <button
                onClick={() => setMessages([])}
                className="text-xs text-gray-600 hover:text-gray-900 px-2 py-1 rounded hover:bg-gray-100"
                title="Mesajları temizle"
              >
                Yeni Sohbet
              </button>
            </div>
          )}
          <div
            ref={messagesContainerRef}
            onScroll={handleMessagesScroll}
            className="flex-1 overflow-y-auto p-4 space-y-3"
          >
            {messages.length === 0 && (
              <p className="text-sm text-gray-400 text-center mt-8">
                {selectedDoc
                  ? "Soru yazıp gönderebilirsin."
                  : "Önce sol panelden bir doküman seç."}
              </p>
            )}
            {messages.map((msg, i) => {
              const isLastStreaming = isStreaming && i === messages.length - 1;
              if (msg.role === "user") {
                return (
                  <div
                    key={i}
                    className="max-w-xl ml-auto bg-blue-500 text-white px-3 py-2 rounded-lg whitespace-pre-wrap"
                  >
                    {msg.content}
                  </div>
                );
              }
              // Asistan
              return (
                <div
                  key={i}
                  className="max-w-xl bg-gray-100 px-3 py-2 rounded-lg"
                >
                  {msg.content ? (
                    <AssistantMessage content={msg.content} />
                  ) : isLastStreaming ? (
                    <span className="text-gray-500">...</span>
                  ) : null}
                </div>
              );
            })}
            <div ref={messagesEndRef} />
          </div>

          <div className="border-t p-4 flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              disabled={isStreaming}
              placeholder={selectedDoc ? "Soru yaz..." : "Önce doküman seç"}
              className="flex-1 border rounded px-3 py-2 disabled:bg-gray-100"
            />
            <button
              onClick={handleSend}
              disabled={!selectedDoc || !input.trim() || isStreaming}
              className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed"
            >
              {isStreaming ? "..." : "Gönder"}
            </button>
          </div>
        </main>
      </div>

      {conflicts.length > 0 && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl max-w-lg w-full mx-4 p-6">
            <h3 className="text-lg font-semibold mb-2">Çakışan Dosyalar</h3>
            <p className="text-sm text-gray-600 mb-4">
              Aşağıdaki dosyalar koleksiyonda zaten var. Her biri için ne
              yapılacağını seç.
            </p>

            <div className="flex gap-2 mb-3 text-xs">
              <button
                onClick={() => setAllDecisions("overwrite")}
                className="px-2 py-1 border rounded hover:bg-gray-50"
              >
                Tümüne üzerine yaz
              </button>
              <button
                onClick={() => setAllDecisions("skip")}
                className="px-2 py-1 border rounded hover:bg-gray-50"
              >
                Tümünü atla
              </button>
            </div>

            <ul className="space-y-2 max-h-64 overflow-y-auto mb-4">
              {conflicts.map((name) => (
                <li
                  key={name}
                  className="flex items-center justify-between gap-2"
                >
                  <span className="text-sm truncate flex-1">{name}</span>
                  <div className="flex gap-1">
                    <button
                      onClick={() =>
                        setDecisions((prev) => ({
                          ...prev,
                          [name]: "overwrite",
                        }))
                      }
                      className={`text-xs px-2 py-1 rounded ${
                        decisions[name] === "overwrite"
                          ? "bg-blue-500 text-white"
                          : "border hover:bg-gray-50"
                      }`}
                    >
                      Üzerine yaz
                    </button>
                    <button
                      onClick={() =>
                        setDecisions((prev) => ({ ...prev, [name]: "skip" }))
                      }
                      className={`text-xs px-2 py-1 rounded ${
                        decisions[name] === "skip"
                          ? "bg-blue-500 text-white"
                          : "border hover:bg-gray-50"
                      }`}
                    >
                      Atla
                    </button>
                  </div>
                </li>
              ))}
            </ul>

            <div className="flex justify-end gap-2">
              <button
                onClick={cancelConflicts}
                className="px-3 py-1 border rounded hover:bg-gray-50"
              >
                İptal
              </button>
              <button
                onClick={confirmConflicts}
                className="px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600"
              >
                Devam Et
              </button>
            </div>
          </div>
        </div>
      )}

      {isUploading && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl px-8 py-6 flex items-center gap-4">
            <div className="w-6 h-6 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
            <div>
              <p className="font-medium">Dokümanlar işleniyor...</p>
              <p className="text-xs text-gray-500 mt-1">
                PDF parse + VLM analizi + embedding. Bu dakikalar sürebilir.
              </p>
            </div>
          </div>
        </div>
      )}

      {toast && (
        <div className="fixed bottom-4 right-4 z-50">
          <div
            className={`px-4 py-3 rounded-lg shadow-lg max-w-md text-sm flex items-start gap-3 ${
              toast.type === "success"
                ? "bg-green-50 border border-green-200 text-green-900"
                : toast.type === "error"
                  ? "bg-red-50 border border-red-200 text-red-900"
                  : "bg-blue-50 border border-blue-200 text-blue-900"
            }`}
          >
            <span className="flex-1">{toast.message}</span>
            <button
              onClick={() => setToast(null)}
              className="text-gray-500 hover:text-gray-700 text-xs"
            >
              ×
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
