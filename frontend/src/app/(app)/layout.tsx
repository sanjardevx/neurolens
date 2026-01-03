import { AppShell } from "@/components/app-shell";
import { SessionGate } from "@/components/session";

export default function AppLayout({ children }: LayoutProps<"/">) {
  return (
    <SessionGate>
      <AppShell>{children}</AppShell>
    </SessionGate>
  );
}
