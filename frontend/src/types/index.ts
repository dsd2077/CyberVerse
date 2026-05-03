// Character image info
export interface ImageInfo {
  filename: string
  orig_name: string
  added_at: string
  url?: string
}

// Character data model
export interface Character {
  id: string
  name: string
  description: string
  avatar_image: string
  idle_video_url?: string
  idle_video_urls?: string[]
  use_face_crop: boolean
  mode: 'standard' | 'voice_llm'
  voice_provider: string
  voice_type: string
  components: CharacterComponents
  speaking_style: string
  personality: string
  welcome_message: string
  system_prompt: string
  tags: string[]
  images: ImageInfo[]
  active_image: string
  image_mode: string
  created_at: string
  updated_at: string
}

export type CharacterForm = Omit<Character, 'id' | 'created_at' | 'updated_at' | 'images' | 'active_image'>

export interface CharacterComponents {
  llm: string
  asr: string
  tts: string
}

export interface ComponentOption {
  id: string
  name: string
  model: string
  default: boolean
  available: boolean
}

export interface ComponentsResponse {
  llm: ComponentOption[]
  asr: ComponentOption[]
  tts: ComponentOption[]
}

// Settings
export interface DoubaoSettings {
  access_token: string
  app_id: string
}

export interface LiveKitSettings {
  url: string
  api_key: string
  api_secret: string
}

export interface ModelProviderSettings {
  dashscope_api_key: string
  openai_api_key: string
}

export interface InferenceSettings {
  grpc_addr: string
}

export interface Settings {
  doubao: DoubaoSettings
  livekit: LiveKitSettings
  model_providers: ModelProviderSettings
  inference: InferenceSettings
}

// Launch config
export interface ConfigParam {
  name: string
  path: string
  value: string | number
  readonly: boolean
  requires_restart: boolean
  options?: string[]
}

export interface ConfigSection {
  title: string
  badge: 'restart' | 'configurable'
  params: ConfigParam[]
  collapsed?: boolean
}

export interface LaunchConfig {
  active_model: string
  configured_default_model: string
  config_status: AvatarModelConfigStatus
  sections: ConfigSection[]
}

export interface LaunchConfigUpdate {
  model: string
  params: Array<{ path: string; value: string | number }>
}

export interface AvatarModelConfigStatus {
  has_infer_params: boolean
  config_sections_available: string[]
}

export interface AvatarModelDescriptor {
  name: string
  display_name: string
  is_active: boolean
  is_configured_default: boolean
  config_status: AvatarModelConfigStatus
}

export interface AvatarModelInfo {
  active_model: string
  configured_default_model: string
  models: AvatarModelDescriptor[]
  config_status: AvatarModelConfigStatus
}

// Voice types
export interface VoiceOption {
  label: string
  value: string
}

// Aliyun Qwen TTS system voices — values match the voice request parameter.
export const QWEN_TTS_VOICE_OPTIONS: VoiceOption[] = [
  { label: '芊悦 (Cherry)', value: 'Cherry' },
  { label: '苏瑶 (Serena)', value: 'Serena' },
  { label: '晨煦 (Ethan)', value: 'Ethan' },
  { label: '千雪 (Chelsie)', value: 'Chelsie' },
  { label: '茉兔 (Momo)', value: 'Momo' },
  { label: '十三 (Vivian)', value: 'Vivian' },
  { label: '月白 (Moon)', value: 'Moon' },
  { label: '四月 (Maia)', value: 'Maia' },
  { label: '凯 (Kai)', value: 'Kai' },
  { label: '不吃鱼 (Nofish)', value: 'Nofish' },
  { label: '萌宝 (Bella)', value: 'Bella' },
  { label: '詹妮弗 (Jennifer)', value: 'Jennifer' },
  { label: '甜茶 (Ryan)', value: 'Ryan' },
  { label: '卡捷琳娜 (Katerina)', value: 'Katerina' },
  { label: '艾登 (Aiden)', value: 'Aiden' },
  { label: '沧明子 (Eldric Sage)', value: 'Eldric Sage' },
  { label: '乖小妹 (Mia)', value: 'Mia' },
  { label: '沙小弥 (Mochi)', value: 'Mochi' },
  { label: '燕铮莺 (Bellona)', value: 'Bellona' },
  { label: '田叔 (Vincent)', value: 'Vincent' },
  { label: '萌小姬 (Bunny)', value: 'Bunny' },
  { label: '阿闻 (Neil)', value: 'Neil' },
  { label: '墨讲师 (Elias)', value: 'Elias' },
  { label: '徐大爷 (Arthur)', value: 'Arthur' },
  { label: '邻家妹妹 (Nini)', value: 'Nini' },
  { label: '小婉 (Seren)', value: 'Seren' },
  { label: '顽屁小孩 (Pip)', value: 'Pip' },
  { label: '少女阿月 (Stella)', value: 'Stella' },
  { label: '博德加 (Bodega)', value: 'Bodega' },
  { label: '索尼莎 (Sonrisa)', value: 'Sonrisa' },
  { label: '阿列克 (Alek)', value: 'Alek' },
  { label: '多尔切 (Dolce)', value: 'Dolce' },
  { label: '素熙 (Sohee)', value: 'Sohee' },
  { label: '小野杏 (Ono Anna)', value: 'Ono Anna' },
  { label: '莱恩 (Lenn)', value: 'Lenn' },
  { label: '埃米尔安 (Emilien)', value: 'Emilien' },
  { label: '安德雷 (Andre)', value: 'Andre' },
  { label: '拉迪奥·戈尔 (Radio Gol)', value: 'Radio Gol' },
  { label: '上海-阿珍 (Jada)', value: 'Jada' },
  { label: '北京-晓东 (Dylan)', value: 'Dylan' },
  { label: '南京-老李 (Li)', value: 'Li' },
  { label: '陕西-秦川 (Marcus)', value: 'Marcus' },
  { label: '闽南-阿杰 (Roy)', value: 'Roy' },
  { label: '天津-李彼得 (Peter)', value: 'Peter' },
  { label: '四川-晴儿 (Sunny)', value: 'Sunny' },
  { label: '四川-程川 (Eric)', value: 'Eric' },
  { label: '粤语-阿强 (Rocky)', value: 'Rocky' },
  { label: '粤语-阿清 (Kiki)', value: 'Kiki' },
]

// Aliyun Qwen3.5 Omni realtime voices — values match the voice request parameter.
export const QWEN_OMNI_VOICE_OPTIONS: VoiceOption[] = [
  { label: '甜甜 (Tina)', value: 'Tina' },
  { label: '林欣宜 (Cindy)', value: 'Cindy' },
  { label: '清欢 (Liora Mira)', value: 'Liora Mira' },
  { label: '知芝 (Sunnybobi)', value: 'Sunnybobi' },
  { label: '林川野 (Raymond)', value: 'Raymond' },
  { label: '晨煦 (Ethan)', value: 'Ethan' },
  { label: '予安 (Theo Calm)', value: 'Theo Calm' },
  { label: '苏瑶 (Serena)', value: 'Serena' },
  { label: '厚 (Harvey)', value: 'Harvey' },
  { label: '四月 (Maia)', value: 'Maia' },
  { label: '江晨 (Evan)', value: 'Evan' },
  { label: '小乔妹 (Qiao)', value: 'Qiao' },
  { label: '茉兔 (Momo)', value: 'Momo' },
  { label: '伟伦 (Wil)', value: 'Wil' },
  { label: '台普 - 安琪 (Angel)', value: 'Angel' },
  { label: '东厂 - 李公公 (Li Cassian)', value: 'Li Cassian' },
  { label: '温柔生活博主 - 舒然 (Mia)', value: 'Mia' },
  { label: '喜剧担当 - 阿逗 (Joyner)', value: 'Joyner' },
  { label: '金爷 (Gold)', value: 'Gold' },
  { label: '卡捷琳娜 (Katerina)', value: 'Katerina' },
  { label: '甜茶 (Ryan)', value: 'Ryan' },
  { label: '詹妮弗 (Jennifer)', value: 'Jennifer' },
  { label: '艾登 (Aiden)', value: 'Aiden' },
  { label: '敏儿 (Mione)', value: 'Mione' },
  { label: '四川 - 晴儿 (Sunny)', value: 'Sunny' },
  { label: '北京 - 晓东 (Dylan)', value: 'Dylan' },
  { label: '四川 - 程川 (Eric)', value: 'Eric' },
  { label: '天津 - 李彼得 (Peter)', value: 'Peter' },
  { label: '阿樸伯 (Joseph Chen)', value: 'Joseph Chen' },
  { label: '陕西 - 秦川 (Marcus)', value: 'Marcus' },
  { label: '南京 - 老李 (Li)', value: 'Li' },
  { label: '粤语 - 阿强 (Rocky)', value: 'Rocky' },
  { label: '素熙 (Sohee)', value: 'Sohee' },
  { label: '莱恩 (Lenn)', value: 'Lenn' },
  { label: '小野杏 (Ono Anna)', value: 'Ono Anna' },
  { label: '索尼莎 (Sonrisa)', value: 'Sonrisa' },
  { label: '博德加 (Bodega)', value: 'Bodega' },
  { label: '埃米尔安 (Emilien)', value: 'Emilien' },
  { label: '安德雷 (Andre)', value: 'Andre' },
  { label: '拉迪奥·戈尔 (Radio Gol)', value: 'Radio Gol' },
  { label: '阿列克 (Alek)', value: 'Alek' },
  { label: '阿力 (Rizky)', value: 'Rizky' },
  { label: '萝雅 (Roya)', value: 'Roya' },
  { label: '阿尔达 (Arda)', value: 'Arda' },
  { label: '阿幸 (Hana)', value: 'Hana' },
  { label: '多尔切 (Dolce)', value: 'Dolce' },
  { label: '雅克 (Jakub)', value: 'Jakub' },
  { label: '海娜 (Griet)', value: 'Griet' },
  { label: '艾莉卡 (Eliška)', value: 'Eliška' },
  { label: '玛丽娜 (Marina)', value: 'Marina' },
  { label: '西芮 (Siiri)', value: 'Siiri' },
  { label: '林恩 (Ingrid)', value: 'Ingrid' },
  { label: '海娜 (Sigga)', value: 'Sigga' },
  { label: '雅娜 (Bea)', value: 'Bea' },
  { label: '思怡 (Chloe)', value: 'Chloe' },
]

// SC2.0 official voices — values match SC20_VOICES keys in doubao_config.py
export const VOICE_OPTIONS: VoiceOption[] = [
  // Female
  { label: '傲娇女友', value: '傲娇女友' },
  { label: '冰娇姐姐', value: '冰娇姐姐' },
  { label: '成熟姐姐', value: '成熟姐姐' },
  { label: '可爱女生', value: '可爱女生' },
  { label: '暖心学姐', value: '暖心学姐' },
  { label: '贴心女友', value: '贴心女友' },
  { label: '温柔文雅', value: '温柔文雅' },
  { label: '妩媚御姐', value: '妩媚御姐' },
  { label: '性感御姐', value: '性感御姐' },
  // Male
  { label: '爱气凌人', value: '爱气凌人' },
  { label: '傲娇公子', value: '傲娇公子' },
  { label: '傲娇精英', value: '傲娇精英' },
  { label: '傲慢少爷', value: '傲慢少爷' },
  { label: '霸道少爷', value: '霸道少爷' },
  { label: '冰娇白莲', value: '冰娇白莲' },
  { label: '不羁青年', value: '不羁青年' },
  { label: '成熟总裁', value: '成熟总裁' },
  { label: '磁性男嗓', value: '磁性男嗓' },
  { label: '醋精男友', value: '醋精男友' },
  { label: '风发少年', value: '风发少年' },
  { label: '腹黑公子', value: '腹黑公子' },
]

export const OPENAI_VOICE_OPTIONS: VoiceOption[] = [
  { label: 'alloy', value: 'alloy' },
  { label: 'ash', value: 'ash' },
  { label: 'ballad', value: 'ballad' },
  { label: 'coral', value: 'coral' },
  { label: 'echo', value: 'echo' },
  { label: 'fable', value: 'fable' },
  { label: 'nova', value: 'nova' },
  { label: 'onyx', value: 'onyx' },
  { label: 'sage', value: 'sage' },
  { label: 'shimmer', value: 'shimmer' },
]
