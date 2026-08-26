import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'

const guardedFiles = [
  'src/pages/AccountPage.tsx',
  'src/pages/ContributorPage.tsx',
  'src/pages/ModelsPage.tsx',
  'src/pages/Playground.tsx',
  'src/pages/JobTimelinePage.tsx',
  'src/pages/JobsPage.tsx',
  'src/components/MeshMap.tsx',
  'src/components/NodeCard.tsx',
]

const allowedBareNumericExpressions: Array<{ file: string; includes: string; reason: string }> = [
  { file: 'src/components/MeshMap.tsx', includes: 'const a  = (Math.sin', reason: 'canvas particle opacity, not rendered as text' },
  { file: 'src/pages/JobTimelinePage.tsx', includes: 'return `${(ms / 1000).toFixed(2)} s`', reason: 'shared formatter used by Metric and trace log rows' },
  { file: 'src/pages/JobsPage.tsx', includes: 'const percent = job.progress.total', reason: 'derived value rendered through Metric' },
]

function isAllowed(file: string, line: string) {
  if (line.includes('<Metric ')) return true
  return allowedBareNumericExpressions.some((entry) => entry.file === file && line.includes(entry.includes))
}

describe('numeric integrity guard', () => {
  it('keeps formatted user-facing numbers behind Metric tier labels or an explicit allowlist', () => {
    const offenders: string[] = []
    for (const file of guardedFiles) {
      const source = readFileSync(join(process.cwd(), file), 'utf8')
      source.split(/\r?\n/).forEach((line, index) => {
        if (!/(\.toFixed\(|Math\.round\(|formatNumber\()/.test(line)) return
        if (!isAllowed(file, line)) offenders.push(`${file}:${index + 1}: ${line.trim()}`)
      })
    }
    expect(offenders).toEqual([])
  })

  it('documents every non-Metric exception with a reason', () => {
    expect(allowedBareNumericExpressions.map((entry) => entry.reason)).toEqual([
      'canvas particle opacity, not rendered as text',
      'shared formatter used by Metric and trace log rows',
      'derived value rendered through Metric',
    ])
  })
})
