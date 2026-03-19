"use client";

import { useBackgroundToasts } from "@/hooks/use-background-toasts";

export function ToastListener() {
  useBackgroundToasts();
  return null;
}
