"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Sparkles, Loader2, Settings2 } from "lucide-react";

interface Props {
  onGenerate: (params: Record<string, unknown>) => void;
  loading: boolean;
}

export function GenerateBar({ onGenerate, loading }: Props) {
  const [count, setCount] = useState(4);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [seed, setSeed] = useState("");
  const [sampler, setSampler] = useState("ddim");
  const [steps, setSteps] = useState(50);
  const [eta, setEta] = useState(0);

  function handleSubmit() {
    const params: Record<string, unknown> = { batch_size: count, sampler, num_steps: steps, eta };
    if (seed.trim()) params.seed = parseInt(seed, 10);
    onGenerate(params);
  }

  return (
    <div className="space-y-2">
      {showAdvanced && (
        <div className="flex flex-wrap items-end gap-4 rounded-xl border border-border bg-card/95 backdrop-blur-xl px-4 py-3 shadow-lg">
          <div className="space-y-1">
            <Label className="text-xs text-muted-foreground">Seed</Label>
            <Input value={seed} onChange={(e) => setSeed(e.target.value)} placeholder="Random" type="number" className="h-9 w-28" />
          </div>
          <div className="space-y-1">
            <Label className="text-xs text-muted-foreground">Sampler</Label>
            <Select value={sampler} onValueChange={(v) => v && setSampler(v)}>
              <SelectTrigger className="h-9 w-24"><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="ddim">DDIM</SelectItem>
                <SelectItem value="ddpm">DDPM</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-1 flex-1 min-w-[120px]">
            <Label className="text-xs text-muted-foreground">Steps {steps}</Label>
            <Slider value={[steps]} onValueChange={(v) => setSteps(Array.isArray(v) ? v[0] : v)} min={10} max={200} step={5} />
          </div>
          <div className="space-y-1 flex-1 min-w-[100px]">
            <Label className="text-xs text-muted-foreground">Eta {eta.toFixed(2)}</Label>
            <Slider value={[eta]} onValueChange={(v) => setEta(Array.isArray(v) ? v[0] : v)} min={0} max={1} step={0.05} />
          </div>
        </div>
      )}

      <div className="flex items-center gap-3 rounded-xl border border-border bg-card/95 backdrop-blur-xl px-4 py-2.5 shadow-lg">
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className={`shrink-0 p-2 rounded-lg transition-colors ${showAdvanced ? "bg-accent text-foreground" : "text-muted-foreground hover:text-foreground"}`}
        >
          <Settings2 className="h-4 w-4" />
        </button>
        <div className="flex items-center gap-3 flex-1">
          <span className="text-xs text-muted-foreground shrink-0">{count}x</span>
          <div className="flex-1 min-w-[80px]">
            <Slider value={[count]} onValueChange={(v) => setCount(Array.isArray(v) ? v[0] : v)} min={1} max={16} step={1} />
          </div>
        </div>
        <Button onClick={handleSubmit} disabled={loading} className="shrink-0">
          {loading ? <Loader2 className="mr-1.5 h-4 w-4 animate-spin" /> : <Sparkles className="mr-1.5 h-4 w-4" />}
          {loading ? "Running" : "Generate"}
        </Button>
      </div>
    </div>
  );
}
