/**
 * Resolve the React Router basename from Vite's BASE_URL.
 *
 * Vite's `BASE_URL` keeps a trailing slash ("/console/") because it is used to
 * prefix asset URLs. React Router's `basename` must not have one: given
 * basename="/console/", the URL "/console" does not match, React Router renders
 * nothing, and the user gets a completely blank page at the exact URL the server
 * advertises (`GET /console`). Stripping the trailing slash makes both
 * "/console" and "/console/" resolve.
 *
 * Kept in its own module (rather than inline in main.tsx) so it can be tested
 * without executing the app's `createRoot` side effect.
 */
export function routerBasename(baseUrl: string): string {
  const trimmed = baseUrl.replace(/\/+$/, '')
  return trimmed === '' ? '/' : trimmed
}
