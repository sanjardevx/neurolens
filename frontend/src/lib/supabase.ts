import { createClient, type SupabaseClient } from "@supabase/supabase-js";

const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
const key = process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY;

/** Browser login client. The publishable key is public by design; the API checks every token and role. */
export const supabase: SupabaseClient | null = url && key ? createClient(url, key) : null;
