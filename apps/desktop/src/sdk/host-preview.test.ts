import { beforeEach, describe, expect, it, vi } from 'vitest'

import { localPreviewTarget } from '@/lib/local-preview'
import type * as LocalPreview from '@/lib/local-preview'
import type * as PreviewStore from '@/store/preview'

const openPreview = vi.hoisted(() => vi.fn())
const normalizeOrLocalPreviewTarget = vi.hoisted(() => vi.fn())

vi.mock('@/store/preview', async importOriginal => ({
  ...(await importOriginal<typeof PreviewStore>()),
  openPreview
}))

vi.mock('@/lib/local-preview', async importOriginal => ({
  ...(await importOriginal<typeof LocalPreview>()),
  normalizeOrLocalPreviewTarget
}))

const { host } = await import('./index')

describe('host.preview', () => {
  beforeEach(() => {
    openPreview.mockClear()
    normalizeOrLocalPreviewTarget.mockReset()
  })

  it('opens a resolvable target in the rail and reports success', async () => {
    const target = { kind: 'file', label: 'audit.md', previewKind: 'text', source: '/w/audit.md', url: 'file:///w/audit.md' }
    normalizeOrLocalPreviewTarget.mockResolvedValue(target)

    await expect(host.preview('/w/audit.md')).resolves.toBe(true)
    expect(openPreview).toHaveBeenCalledWith(target, 'tool-result')
  })

  it('reports failure instead of opening an empty tab when the target will not resolve', async () => {
    normalizeOrLocalPreviewTarget.mockResolvedValue(null)

    // The caller needs a false here to fall back (kanban drops to a download);
    // opening a tab that renders an error would strand the user.
    await expect(host.preview('/w/missing.md')).resolves.toBe(false)
    expect(openPreview).not.toHaveBeenCalled()
  })

  it('refuses a binary target rather than rendering garbage', async () => {
    normalizeOrLocalPreviewTarget.mockResolvedValue({
      kind: 'file',
      label: 'bundle.zip',
      previewKind: 'binary',
      source: '/w/bundle.zip',
      url: 'file:///w/bundle.zip'
    })

    await expect(host.preview('/w/bundle.zip')).resolves.toBe(false)
    expect(openPreview).not.toHaveBeenCalled()
  })

  it('passes an explicit cwd through for relative targets', async () => {
    normalizeOrLocalPreviewTarget.mockResolvedValue(null)

    await host.preview('docs/audit.md', '/repo')
    expect(normalizeOrLocalPreviewTarget).toHaveBeenCalledWith('docs/audit.md', '/repo')
  })
})

describe('localPreviewTarget classification for agent artifacts', () => {
  it('treats a markdown artifact as readable text, not a download', () => {
    // The reported symptom: an agent writes an audit .md, and the only way to
    // read it was to find it on disk. It must classify as previewable text.
    const target = localPreviewTarget('/w/kanban/attachments/t_1/audit-report.md')

    expect(target).toMatchObject({ kind: 'file', label: 'audit-report.md', language: 'markdown', previewKind: 'text' })
  })

  it('keeps images and html on their own preview kinds', () => {
    expect(localPreviewTarget('/w/shot.png')).toMatchObject({ previewKind: 'image' })
    expect(localPreviewTarget('/w/report.html')).toMatchObject({ previewKind: 'html' })
  })
})
