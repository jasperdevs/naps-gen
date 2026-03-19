"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/lib/api";
import { Archive, Trash2, Loader2 } from "lucide-react";
import { toast } from "sonner";

interface StorageInfo {
  dataset: { size_mb: number; images: number };
  checkpoints: { size_mb: number };
  outputs: { size_mb: number };
  samples: { size_mb: number };
  archives: { size_mb: number };
}

export default function SettingsPage() {
  const [storage, setStorage] = useState<StorageInfo | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);

  function loadStorage() {
    api.getStorage().then((d) => setStorage(d as StorageInfo)).catch(() => {}).finally(() => setLoaded(true));
  }
  useEffect(() => { loadStorage(); }, []);

  async function handleArchive(target: string) {
    setBusy(`archive-${target}`);
    try { await api.archive(target); toast.success(`Archived ${target}`); loadStorage(); }
    catch { toast.error("Failed"); }
    finally { setBusy(null); }
  }

  async function handleDelete(target: string) {
    setBusy(`delete-${target}`);
    try { await api.deleteData(target); toast.success(`Deleted ${target}`); loadStorage(); }
    catch { toast.error("Failed"); }
    finally { setBusy(null); }
  }

  async function handleDeleteArchives() {
    setBusy("delete-archives");
    try {
      const res = await api.deleteArchives() as { deleted: number; freed_mb: number };
      toast.success(`Deleted ${res.deleted} archives (${res.freed_mb} MB freed)`);
      loadStorage();
    } catch { toast.error("Failed"); }
    finally { setBusy(null); }
  }

  return (
    <div className="space-y-6">
      <h1 className="text-lg font-semibold">Settings</h1>

      <Card size="sm">
        <CardHeader>
          <CardTitle className="text-sm font-medium">Storage</CardTitle>
        </CardHeader>
        <CardContent>
          {!loaded ? (
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-5">
              {Array.from({ length: 5 }).map((_, i) => <Skeleton key={i} className="h-10" />)}
            </div>
          ) : storage ? (
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-5">
              <StorageItem label="Dataset" value={`${storage.dataset.size_mb} MB`} />
              <StorageItem label="Checkpoints" value={`${storage.checkpoints.size_mb} MB`} />
              <StorageItem label="Outputs" value={`${storage.outputs.size_mb} MB`} />
              <StorageItem label="Samples" value={`${storage.samples.size_mb} MB`} />
              <StorageItem label="Archives" value={`${storage.archives.size_mb} MB`} />
            </div>
          ) : null}
        </CardContent>
      </Card>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <ActionCard title="Dataset" target="dataset" busy={busy} onArchive={handleArchive} onDelete={handleDelete} />
        <ActionCard title="Training" target="training_history" busy={busy} onArchive={handleArchive} onDelete={handleDelete} />
        <ActionCard title="Generated" target="outputs" busy={busy} onArchive={handleArchive} onDelete={handleDelete} />
        <ActionCard title="Everything" target="all" busy={busy} onArchive={handleArchive} onDelete={handleDelete} isDanger />
      </div>

      <Card size="sm">
        <CardHeader>
          <CardTitle className="text-sm font-medium">Archives</CardTitle>
        </CardHeader>
        <CardContent>
          <Button variant="destructive" className="w-full" disabled={busy !== null} onClick={handleDeleteArchives}>
            {busy === "delete-archives" ? <Loader2 className="mr-1.5 h-4 w-4 animate-spin" /> : <Trash2 className="mr-1.5 h-4 w-4" />}
            Delete All Archives
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}

function StorageItem({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="text-sm font-semibold">{value}</p>
    </div>
  );
}

function ActionCard({ title, target, busy, onArchive, onDelete, isDanger }: {
  title: string; target: string; busy: string | null;
  onArchive: (t: string) => void; onDelete: (t: string) => void; isDanger?: boolean;
}) {
  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
      </CardHeader>
      <CardContent className="flex gap-2">
        <Button variant="secondary" size="sm" className="flex-1" disabled={busy !== null} onClick={() => onArchive(target)}>
          {busy === `archive-${target}` ? <Loader2 className="mr-1.5 h-4 w-4 animate-spin" /> : <Archive className="mr-1.5 h-4 w-4" />}
          Archive
        </Button>
        <Button variant={isDanger ? "destructive" : "secondary"} size="sm" className="flex-1" disabled={busy !== null} onClick={() => onDelete(target)}>
          {busy === `delete-${target}` ? <Loader2 className="mr-1.5 h-4 w-4 animate-spin" /> : <Trash2 className="mr-1.5 h-4 w-4" />}
          Delete
        </Button>
      </CardContent>
    </Card>
  );
}
