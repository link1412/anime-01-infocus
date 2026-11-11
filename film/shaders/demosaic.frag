#version 410
// Malvar-He-Cutler demosaic, RGGB
out vec4 o;
uniform sampler2D uRaw;
float S(ivec2 p){ return texelFetch(uRaw,p,0).r; }
void main(){
  ivec2 p=ivec2(gl_FragCoord.xy);
  float C=S(p);
  float cross_=S(p+ivec2(0,-1))+S(p+ivec2(0,1))+S(p+ivec2(-1,0))+S(p+ivec2(1,0));
  float diag=S(p+ivec2(-1,-1))+S(p+ivec2(1,-1))+S(p+ivec2(-1,1))+S(p+ivec2(1,1));
  float hh=S(p+ivec2(-2,0))+S(p+ivec2(2,0));
  float vv=S(p+ivec2(0,-2))+S(p+ivec2(0,2));
  float hN=S(p+ivec2(-1,0))+S(p+ivec2(1,0)), vN=S(p+ivec2(0,-1))+S(p+ivec2(0,1));
  float axial=hh+vv;
  float G_at_RB=(4.*C+2.*cross_-axial)/8.;
  float RB_at_G_row=(5.*C+4.*hN-(diag)-hh+.5*vv)/8.;
  float RB_at_G_col=(5.*C+4.*vN-(diag)-vv+.5*hh)/8.;
  float RB_at_BR=(6.*C+2.*diag-1.5*axial)/8.;
  bool ex=(p.x&1)==0, ey=(p.y&1)==0;
  vec3 c;
  if(ex&&ey)        c=vec3(C,G_at_RB,RB_at_BR);
  else if(!ex&&!ey) c=vec3(RB_at_BR,G_at_RB,C);
  else if(!ex&&ey)  c=vec3(RB_at_G_row,C,RB_at_G_col);
  else              c=vec3(RB_at_G_col,C,RB_at_G_row);
  o=vec4(max(c,0.),1.);
}
