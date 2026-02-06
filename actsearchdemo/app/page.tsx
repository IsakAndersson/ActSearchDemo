"use client";

import { FormEvent, useState } from "react";
import { useRouter } from "next/navigation";

const DEMO_PASSWORD = "actsearchdemo";
const SESSION_KEY = "actsearch-authenticated";

export default function LoginPage() {
  const router = useRouter();
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);

  const onSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (password !== DEMO_PASSWORD) {
      setError("Wrong password.");
      return;
    }

    localStorage.setItem(SESSION_KEY, "true");
    router.push("/search");
  };

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top_left,#fff9ef,#f6f0e5)] px-6 py-20 text-[#211f1b]">
      <main className="mx-auto w-full max-w-lg rounded-3xl border border-[#d9cfbf] bg-[#fffcf6] p-10 shadow-[0_20px_60px_rgba(0,0,0,0.08)]">
        <h1 className="mb-2 font-serif text-4xl">ActSearch Demo</h1>
        <p className="mb-8 text-sm text-[#6f6557]">
          Enter the hardcoded password to open the search interface.
        </p>

        <form className="space-y-5" onSubmit={onSubmit}>
          <label className="block">
            <span className="mb-2 block text-sm text-[#6f6557]">Password</span>
            <input
              className="w-full rounded-xl border border-[#cdbfa8] bg-white px-4 py-3 outline-none focus:border-[#1f6e6e]"
              type="password"
              value={password}
              onChange={(event) => {
                setError(null);
                setPassword(event.target.value);
              }}
              placeholder="Password"
              required
            />
          </label>

          {error ? (
            <p className="rounded-lg border border-[#f0b79f] bg-[#ffe8dc] px-3 py-2 text-sm text-[#7a2e0d]">
              {error}
            </p>
          ) : null}

          <button
            className="w-full rounded-xl bg-[#1f6e6e] px-4 py-3 font-semibold text-white transition hover:bg-[#175959]"
            type="submit"
          >
            Continue
          </button>
        </form>
      </main>
    </div>
  );
}
