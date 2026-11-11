#version 410
// Flying white. A fast pan is a dry brush dragged across the paper: every bristle row carries
// ink a different distance, and its track breaks where the bristle has run dry and the paper shows.
#include "common.glsl"
in vec2 uv; out vec4 o;
uniform sampler2D uD,uNoise;
uniform float uAspect,uZoom,uPhPaper,uDir,uLen,uAmt,uStage;
uniform vec2 uZc;
const int N=28;
void main(){
  if(uStage<.5){
    // stage one: spread the ink along the drag just enough that the samples of stage two meet
    vec4 a=vec4(0.);
    for(int i=0;i<10;i++) a+=texture(uD,uv+vec2((float(i)/9.-.5)*uLen*1.25/float(N)/uAspect,0.));
    o=a/10.;
    return;
  }
  vec2 p=vec2((uv.x*2.-1.)*uAspect,uv.y*2.-1.);
  vec2 v=uZc*(1.-1./uZoom)+p/uZoom;
  float y=v.y, X=v.x+uPhPaper;
  // rows of bristles: fixed in y, slowly varying along the paper
  float hair=texture(uNoise,vec2(X*.018,y*.56)).a;
  float hair2=texture(uNoise,vec2(X*.011+.37,y*1.25)).a;
  float band=texture(uNoise,vec2(X*.012+.13,y*.38)).g;
  float band2=texture(uNoise,vec2(X*.006+.71,y*.11)).g;
  float tooth=texture(uNoise,vec2(X*.66,y*.66)).b;
  // how far this bristle dragged its ink
  float L=uLen*(.25+.95*band*band2*2.+.30*hair);
  vec3 acc=vec3(0.),mx=vec3(0.); float ws=0.;
  // the sample offset is fixed to the paper, so that no copies of the image line up
  float jit=texture(uNoise,vec2(X*.31,y*1.3)).a;
  for(int i=0;i<N;i++){
    float s=(float(i)+jit)/float(N);
    vec3 d=texture(uD,uv+vec2(-uDir*s*L/(2.*uAspect),0.)).rgb;
    float w=1.-s*.72;
    acc+=d*w; ws+=w;
    mx=max(mx,d*(1.-s*s*.75));
  }
  vec3 here=texture(uD,uv).rgb;
  vec3 drag=mix(acc/ws,mx,.62);
  // dry: fraction of paper that shows through, rising with speed
  float grip=hair*.50+hair2*.16+band*.14+tooth*.20;
  float th=mix(0.,.54,uAmt);
  float cov=smoothstep(th-.055,th+.055,grip);
  // a bristle that has just been loaded (dense ink) lays more
  float dens=clamp((drag.r+drag.g+drag.b)*.9,0.,1.);
  cov=mix(cov,smoothstep(th-.2,th-.02,grip),dens*.45);
  o=vec4(mix(here,drag*cov*(1.+.35*uAmt),smoothstep(0.,.25,uAmt)),texture(uD,uv).a);
}
