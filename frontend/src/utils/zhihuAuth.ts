export const ZHIHU_OAUTH_STATE_KEY = 'cyberverse_zhihu_oauth_state'
export const ZHIHU_AFTER_AUTH_KEY = 'cyberverse_zhihu_after_auth'

export function getZhihuRedirectUri(): string {
  return `${window.location.origin}/kanshan/oauth/callback`
}
