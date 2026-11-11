#version 410
// One step of conductivity-limited diffusion: r=ink, g=vermilion, b=indigo, a=conductivity.
// Conservative: what leaves one texel arrives at its neighbour. Mobility vanishes on dry paper,
// so the wet front is sharp and can only advance from ink that is already there (a porous-medium
// law). With a conductivity map of fibres, the front runs out along them as feathered fingers.
in vec2 uv; out vec4 o;
uniform sampler2D uS;
uniform vec2 uPx;
uniform float uStep,uDt,uMob;
const vec2 D8[8]=vec2[8](vec2(1,0),vec2(-1,0),vec2(0,1),vec2(0,-1),vec2(1,1),vec2(-1,1),vec2(1,-1),vec2(-1,-1));
const float W8[8]=float[8](1.,1.,1.,1.,.5,.5,.5,.5);
void main(){
  vec4 c=texture(uS,uv);
  float tc=c.r+c.g+c.b;
  vec3 f=vec3(0.);
  for(int i=0;i<8;i++){
    vec4 n=texture(uS,uv+D8[i]*uPx*uStep);
    float tn=n.r+n.g+n.b;
    float g=sqrt(max(c.a*n.a,0.))*W8[i]*smoothstep(0.,uMob,max(tc,tn));
    f+=g*(n.rgb-c.rgb);
  }
  o=vec4(max(c.rgb+uDt*f,0.),c.a);
}
