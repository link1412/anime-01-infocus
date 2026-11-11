#version 410
// What the glass holds: someone, and one solid thing, the camera.
in vec2 uv; out vec4 o;
const float CAM_GAIN=2.2;
uniform sampler2D uForm,uCam;
uniform vec3 uCool;
uniform float uAspect,uBreath,uK;
void main(){
  vec4 t=texture(uForm,uv+vec2(0.,uBreath*.0018));
  vec3 c=uCool*uK*t.r;
  float a=t.g;
  vec4 cam=texture(uCam,uv);
  c=mix(c,cam.rgb*CAM_GAIN,cam.a);
  a=max(a,cam.a);
  o=vec4(c,a);
}
