#version 410
in vec2 uv; out vec4 o;
uniform sampler2D uHero,uSecond,uFar,uBeads,uBeads2,uDrop;
void main(){
  vec4 h=texture(uHero,uv);
  vec4 s=texture(uSecond,uv);
  vec3 far=texture(uFar,uv).rgb;
  vec3 c=s.rgb+far*(1.-clamp(s.a,0.,1.));
  vec4 b2=texture(uBeads2,uv);
  c=b2.rgb+c*(1.-clamp(b2.a,0.,1.));
  c=h.rgb+c*(1.-clamp(h.a,0.,1.));
  vec4 b=texture(uBeads,uv);
  c=b.rgb+c*(1.-clamp(b.a,0.,1.));
  vec4 d=texture(uDrop,uv);
  o=vec4(d.rgb+c*(1.-clamp(d.a,0.,1.)),1.);
}
