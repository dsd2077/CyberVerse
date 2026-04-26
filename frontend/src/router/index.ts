import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(),
  scrollBehavior(_to, _from, savedPosition) {
    if (savedPosition) {
      return savedPosition
    }

    return { left: 0, top: 0 }
  },
  routes: [
    {
      path: '/',
      name: 'landing',
      component: () => import('../pages/LandingPage.vue'),
    },
    {
      path: '/characters',
      name: 'characters',
      component: () => import('../pages/CharacterListPage.vue'),
    },
    {
      path: '/characters/new',
      name: 'character-create',
      component: () => import('../pages/CharacterEditPage.vue'),
    },
    {
      path: '/characters/:id/edit',
      name: 'character-edit',
      component: () => import('../pages/CharacterEditPage.vue'),
    },
    {
      path: '/launch/:id',
      name: 'launch',
      component: () => import('../pages/LaunchConfigPage.vue'),
    },
    {
      path: '/session/:id',
      name: 'session',
      component: () => import('../pages/SessionPage.vue'),
    },
    {
      path: '/settings',
      name: 'settings',
      component: () => import('../pages/SettingsPage.vue'),
    },
  ],
})

export default router
